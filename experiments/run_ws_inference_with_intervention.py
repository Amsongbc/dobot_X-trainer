#!/usr/bin/env python3
"""WebSocket policy inference with per-arm human intervention.

Hold the recording button on either leader hand to take over the matching
follower arm. Release it to return that arm to the server-side policy.
"""

import os
import sys
import time
from dataclasses import dataclass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import cv2
import numpy as np
import tyro

from dobot_control.agents.agent import BimanualAgent
from dobot_control.agents.dobot_agent import DobotAgent
from scripts.manipulate_utils import load_ini_data_hands

import experiments.run_ws_inference as ws


@dataclass
class Args(ws.Args):
    # Per-control-cycle limits. They smooth both manual motion and policy return.
    max_joint_step: float = 0.08
    max_gripper_step: float = 0.10


def make_leader_agent() -> BimanualAgent:
    _, hands = load_ini_data_hands()
    return BimanualAgent(
        DobotAgent(which_hand="LEFT", dobot_config=hands["HAND_LEFT"]),
        DobotAgent(which_hand="RIGHT", dobot_config=hands["HAND_RIGHT"]),
    )


def limit_command(target: np.ndarray, previous: np.ndarray, args: Args) -> np.ndarray:
    result = target.copy()
    joint_indices = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12])
    gripper_indices = np.array([6, 13])
    result[joint_indices] = previous[joint_indices] + np.clip(
        result[joint_indices] - previous[joint_indices],
        -args.max_joint_step,
        args.max_joint_step,
    )
    result[gripper_indices] = previous[gripper_indices] + np.clip(
        result[gripper_indices] - previous[gripper_indices],
        -args.max_gripper_step,
        args.max_gripper_step,
    )
    result[gripper_indices] = np.clip(result[gripper_indices], 0.0, 1.0)
    return result


def target_is_safe(action: np.ndarray) -> bool:
    left_ok = -2.6 < action[2] < 0.0 and action[3] > -0.6
    right_ok = 0.0 < action[9] < 2.6 and action[10] < 0.6
    return left_ok and right_ok


def workspace_is_safe(env) -> bool:
    pos = env.get_XYZrxryrz_state()
    left_ok = -410 < pos[0] < 300 and -700 < pos[1] < -210 and pos[2] > 42
    right_ok = -250 < pos[6] < 410 and -700 < pos[7] < -210 and pos[8] > 42
    return left_ok and right_ok


def set_error_light(env) -> None:
    env.set_do_status([3, 0])
    env.set_do_status([2, 0])
    env.set_do_status([1, 1])


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
    if args.max_joint_step <= 0 or args.max_gripper_step <= 0:
        raise ValueError("manual/policy command step limits must be positive")

    ws_client = None
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
        first = True
        interval = 1.0 / args.control_hz

        was_intervening = np.array([False, False])
        leader_origin = np.zeros(14)
        follower_origin = np.zeros(14)
        previous_chunk = None

        if args.record_video:
            video_path = ws.make_video_path(args.video_path, args.video_dir)
            print("Recording camera video to:", video_path)

        print("WebSocket policy started.")
        print("Hold a leader recording button to take over that arm; release to resume policy.")

        while t < args.episode_len:
            images = ws.get_current_images()
            if args.show_img:
                cv2.imshow("ws policy / intervention", np.hstack(images))
                cv2.waitKey(1)

            request_start = time.time()
            actions = ws_client.infer(args.instruction, images, qpos)
            actions = np.asarray([ws.clamp_grippers(a) for a in actions], dtype=np.float64)
            actions, previous_chunk, server_len, phase = select_actions(
                args, actions, previous_chunk
            )
            print(
                "Server inference time(ms):",
                (time.time() - request_start) * 1000,
                "execute chunk:", len(actions),
                "server chunk:", server_len,
                "ensemble phase:", phase,
            )

            request_fresh_policy = False
            for chunk_idx, policy_action in enumerate(actions):
                if t >= args.episode_len:
                    break
                step_start = time.time()

                keys = leader.get_keys()
                # Hardware buttons are active-low. Column 1 is the recording button.
                intervening = keys[:, 1] == 0
                leader_now = leader.act({})
                actual_now = env.get_obs()["joint_positions"]
                just_released = np.array([False, False])

                for side in range(2):
                    side_slice = slice(side * 7, side * 7 + 7)
                    side_name = "LEFT" if side == 0 else "RIGHT"
                    if intervening[side] and not was_intervening[side]:
                        leader.set_torque(side, False)
                        leader_now = leader.act({})
                        leader_origin[side_slice] = leader_now[side_slice]
                        follower_origin[side_slice] = actual_now[side_slice]
                        print(f"{side_name} manual takeover")
                    elif not intervening[side] and was_intervening[side]:
                        leader.set_torque(side, True)
                        # Remaining actions were inferred before manual motion. Drop
                        # them and ask the server again using the current robot state.
                        request_fresh_policy = True
                        just_released[side] = True
                        print(f"{side_name} returned to policy; discard stale action chunk")

                target = policy_action.copy()
                for side in range(2):
                    side_slice = slice(side * 7, side * 7 + 7)
                    if intervening[side]:
                        target[side_slice] = (
                            follower_origin[side_slice]
                            + leader_now[side_slice]
                            - leader_origin[side_slice]
                        )
                    elif just_released[side]:
                        # Hold this arm for the release cycle. The next outer
                        # iteration requests a fresh action from the server.
                        target[side_slice] = actual_now[side_slice]
                target = ws.clamp_grippers(target)
                command = limit_command(target, last_command, args)

                if not args.dry_run:
                    if not target_is_safe(command) or not workspace_is_safe(env):
                        set_error_light(env)
                        raise RuntimeError("Safety boundary reached; robot command stopped")
                    if first:
                        ws.move_linearly(env, last_command, command, max_steps=100)
                        first = False
                    obs = env.step(command, np.array([1, 1]))
                else:
                    first = False
                    obs = env.get_obs()

                obs["joint_positions"][[6, 13]] = command[[6, 13]]
                qpos = obs["joint_positions"].copy()
                last_command = command.copy()
                was_intervening = intervening.copy()
                t += 1

                if args.record_video:
                    frame = np.hstack(ws.get_current_images())
                    if video_writer is None:
                        assert video_path is not None
                        video_writer = ws.make_video_writer(
                            video_path, frame, args.video_fps or args.control_hz
                        )
                    video_writer.write(frame)

                elapsed = time.time() - step_start
                print("step:", t, "chunk_idx:", chunk_idx, "step ms:", elapsed * 1000)
                if interval > elapsed:
                    time.sleep(interval - elapsed)

                if request_fresh_policy:
                    previous_chunk = None
                    break

        return 0
    finally:
        ws.thread_run = False
        if leader is not None:
            leader.set_torque(2, True)
        if video_writer is not None:
            video_writer.release()
            if video_path is not None:
                print("Saved camera video to:", video_path)
        if ws_client is not None:
            ws_client.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main(tyro.cli(Args)))
