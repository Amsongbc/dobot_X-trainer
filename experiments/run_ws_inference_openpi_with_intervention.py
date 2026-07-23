#!/usr/bin/env python3
"""WebSocket policy inference with per-arm human intervention.

Hold the recording button on either leader hand to take over the matching
follower arm. Release it to return that arm to the server-side policy.
"""

import os
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import cv2
import numpy as np
import tyro

from dobot_control.agents.agent import BimanualAgent
from dobot_control.agents.dobot_agent import DobotAgent
from scripts.manipulate_utils import load_ini_data_hands

import experiments.run_ws_inference_openpi as ws


@dataclass
class Args(ws.Args):
    pass


def make_leader_agent() -> BimanualAgent:
    _, hands = load_ini_data_hands()
    return BimanualAgent(
        DobotAgent(which_hand="LEFT", dobot_config=hands["HAND_LEFT"]),
        DobotAgent(which_hand="RIGHT", dobot_config=hands["HAND_RIGHT"]),
    )


def select_actions(args: Args, actions: np.ndarray, previous_chunk):
    """Apply the same chunk/temporal-ensemble behavior as run_ws_inference."""
    server_chunk_len = len(actions)
    if args.temporal_ensemble:
        effective_len = args.action_chunk_len or server_chunk_len
        if effective_len > server_chunk_len:
            raise ValueError(
                "action_chunk_len cannot exceed server chunk length: "
                f"{effective_len} > {server_chunk_len}"
            )
        actions = actions[:effective_len]
        stride = max(1, int(effective_len / args.ensemble_stride_divisor))
        if stride >= effective_len:
            raise ValueError("temporal ensemble stride must be smaller than chunk length")
        if previous_chunk is None:
            selected = actions[:stride]
            phase = "warmup"
        else:
            overlap = min(stride, len(previous_chunk) - stride, len(actions))
            if overlap <= 0:
                raise RuntimeError("No overlapping actions available for temporal ensemble")
            selected = ws.blend_overlapping_actions(
                previous_chunk[stride:stride + overlap],
                actions[:overlap],
                args.ensemble_temperature,
            )
            phase = "overlap"
        return selected, actions.copy(), server_chunk_len, phase
    if args.action_chunk_len is not None:
        actions = actions[:args.action_chunk_len]
    return actions, previous_chunk, server_chunk_len, None


@dataclass
class InferenceRequest:
    images: list[np.ndarray]
    qpos: np.ndarray
    generation: int


@dataclass
class InferenceResult:
    actions: Optional[np.ndarray]
    generation: int
    elapsed_ms: float
    error: Optional[BaseException] = None


class AsyncInference:
    """Run blocking WebSocket inference without stopping robot teleoperation."""

    def __init__(self, client, instruction: Optional[str]) -> None:
        self.client = client
        self.instruction = instruction
        self._condition = threading.Condition()
        self._request: Optional[InferenceRequest] = None
        self._result: Optional[InferenceResult] = None
        self._busy = False
        self._stopping = False
        self._thread = threading.Thread(
            target=self._run,
            name="ws-policy-inference",
            daemon=True,
        )
        self._thread.start()

    def submit(
        self,
        images: list[np.ndarray],
        qpos: np.ndarray,
        generation: int,
    ) -> bool:
        with self._condition:
            if (
                self._stopping
                or self._busy
                or self._request is not None
                or self._result is not None
            ):
                return False
            self._request = InferenceRequest(
                images=[image.copy() for image in images],
                qpos=qpos.copy(),
                generation=generation,
            )
            self._busy = True
            self._condition.notify_all()
            return True

    def poll(self) -> Optional[InferenceResult]:
        with self._condition:
            result = self._result
            self._result = None
            return result

    def close(self) -> None:
        with self._condition:
            self._stopping = True
            self._condition.notify_all()
        self._thread.join(timeout=5.0)

    def _run(self) -> None:
        while True:
            with self._condition:
                while self._request is None and not self._stopping:
                    self._condition.wait()
                if self._stopping:
                    return
                request, self._request = self._request, None
            assert request is not None
            started = time.monotonic()
            try:
                actions = self.client.infer(
                    self.instruction,
                    request.images,
                    request.qpos,
                )
                result = InferenceResult(
                    actions=np.asarray(actions, dtype=np.float64),
                    generation=request.generation,
                    elapsed_ms=(time.monotonic() - started) * 1000.0,
                )
            except BaseException as exc:
                result = InferenceResult(
                    actions=None,
                    generation=request.generation,
                    elapsed_ms=(time.monotonic() - started) * 1000.0,
                    error=exc,
                )
            with self._condition:
                self._result = result
                self._busy = False
                self._condition.notify_all()


def main(args: Args) -> int:
    if args.action_chunk_len is not None and args.action_chunk_len <= 0:
        raise ValueError("action_chunk_len must be positive")
    if args.control_hz <= 0:
        raise ValueError("control_hz must be positive")
    if args.ensemble_temperature <= 0:
        raise ValueError("ensemble_temperature must be positive")
    if args.ensemble_stride_divisor <= 1:
        raise ValueError("ensemble_stride_divisor must be greater than 1")
    if args.video_fps is not None and args.video_fps <= 0:
        raise ValueError("video_fps must be positive when set")
    ws_client = None
    inference = None
    leader = None
    video_writer = None
    video_path = None
    ws.thread_run = False

    try:
        if args.check_only:
            # Delegate the server health check; no robot, cameras, or leader are opened.
            return ws.main(args)

        ws_client = ws.OpenPIWebSocketClient(args.ws_host, args.ws_port)
        print("OpenPI server metadata:", ws_client.metadata())
        inference = AsyncInference(ws_client, args.instruction)
        ws.init_cameras(crop_top_camera=args.crop_top_camera)
        env = ws.init_robot(args)

        leader = make_leader_agent()
        # The two leader hands stay locked until their recording buttons are held.
        leader.set_torque(2, True)

        obs = env.get_obs()
        obs["joint_positions"][[6, 13]] = 1.0
        qpos = obs["joint_positions"].copy()
        last_command = qpos.copy()
        t = 0
        interval = 1.0 / args.control_hz

        was_intervening = np.array([False, False])
        leader_origin = np.zeros(14)
        follower_origin = np.zeros(14)
        previous_chunk = None
        policy_actions: deque[np.ndarray] = deque()
        policy_generation = 0

        if args.record_video:
            video_path = ws.make_video_path(args.video_path, args.video_dir)
            print("Recording camera video to:", video_path)

        print("WebSocket policy started.")
        print("Hold a leader recording button to take over that arm; release to resume policy.")

        while t < args.episode_len:
            step_start = time.monotonic()
            images = ws.get_current_images()
            if args.show_img:
                cv2.imshow("ws policy / intervention", np.hstack(images))
                cv2.waitKey(1)

            result = inference.poll()
            if result is not None:
                if result.error is not None:
                    raise result.error
                if result.generation != policy_generation or was_intervening.any():
                    previous_chunk = None
                    print(
                        "Discard stale policy returned during intervention:",
                        f"request_generation={result.generation}",
                        f"current_generation={policy_generation}",
                    )
                else:
                    assert result.actions is not None
                    raw_actions = np.asarray(
                        [ws.clamp_grippers(action) for action in result.actions],
                        dtype=np.float64,
                    )
                    selected, previous_chunk, server_len, phase = select_actions(
                        args,
                        raw_actions,
                        previous_chunk,
                    )
                    policy_actions.extend(selected)
                    print(
                        "Server inference time(ms):",
                        result.elapsed_ms,
                        "execute chunk:",
                        len(selected),
                        "server chunk:",
                        server_len,
                        "ensemble phase:",
                        phase,
                    )

            keys = leader.get_keys()
            # Hardware buttons are active-low. Column 1 is the recording button.
            intervening = np.asarray(keys[:, 1] == 0, dtype=bool)
            leader_now = leader.act({})
            just_released = np.zeros(2, dtype=bool)

            for side in range(2):
                side_slice = slice(side * 7, side * 7 + 7)
                side_name = "LEFT" if side == 0 else "RIGHT"
                if intervening[side] and not was_intervening[side]:
                    leader.set_torque(side, False)
                    leader_now = leader.act({})
                    leader_origin[side_slice] = leader_now[side_slice]
                    follower_origin[side_slice] = last_command[side_slice]
                    policy_actions.clear()
                    previous_chunk = None
                    policy_generation += 1
                    print(f"{side_name} manual takeover; discard stale policy")
                elif not intervening[side] and was_intervening[side]:
                    leader.set_torque(side, True)
                    just_released[side] = True
                    policy_actions.clear()
                    previous_chunk = None
                    policy_generation += 1
                    print(f"{side_name} returned to policy; request fresh policy")

            should_execute = False
            target = last_command.copy()
            if intervening.any():
                should_execute = True
                for side in range(2):
                    side_slice = slice(side * 7, side * 7 + 7)
                    if intervening[side]:
                        target[side_slice] = (
                            follower_origin[side_slice]
                            + leader_now[side_slice]
                            - leader_origin[side_slice]
                        )
            elif just_released.any():
                # Execute one hold cycle, then infer from the post-intervention pose.
                should_execute = True
            elif policy_actions:
                target = policy_actions.popleft()
                should_execute = True

            if should_execute:
                target = np.asarray(target, dtype=np.float64)
                if target.shape != (14,) or not np.isfinite(target).all():
                    raise RuntimeError(
                        f"Invalid command: shape={target.shape}, "
                        f"finite={np.isfinite(target).all()}"
                    )
                command = ws.clamp_grippers(target)
                if not args.dry_run:
                    obs = env.step(command, np.array([1, 1]))
                else:
                    obs = env.get_obs()

                obs["joint_positions"][[6, 13]] = command[[6, 13]]
                qpos = obs["joint_positions"].copy()
                last_command = command.copy()
                t += 1

                if args.record_video:
                    frame = np.hstack(images)
                    if video_writer is None:
                        assert video_path is not None
                        video_writer = ws.make_video_writer(
                            video_path,
                            frame,
                            args.video_fps or args.control_hz,
                        )
                    video_writer.write(frame)
                print(
                    "step:",
                    t,
                    "policy_buffer:",
                    len(policy_actions),
                    "manual:",
                    intervening.tolist(),
                )

            was_intervening = intervening.copy()

            # Request policy only from a stable, non-intervened state. The
            # blocking request runs in AsyncInference while this loop keeps
            # polling and executing manual commands at control_hz.
            if not policy_actions and not intervening.any():
                inference.submit(images, qpos, policy_generation)

            remaining = interval - (time.monotonic() - step_start)
            if remaining > 0:
                time.sleep(remaining)

        return 0
    finally:
        ws.thread_run = False
        if leader is not None:
            leader.set_torque(2, True)
        if video_writer is not None:
            video_writer.release()
            if video_path is not None:
                print("Saved camera video to:", video_path)
        if inference is not None:
            inference.close()
        if ws_client is not None:
            ws_client.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main(tyro.cli(Args)))
