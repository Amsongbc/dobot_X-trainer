#!/usr/bin/env python3
import base64
import json
import os
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import cv2
import numpy as np
import tyro

from dobot_control.cameras.realsense_camera import RealSenseCamera
from dobot_control.env import RobotEnv
from dobot_control.robots.robot_node import ZMQClientRobot
from scripts.manipulate_utils import load_ini_data_camera

try:
    from websocket import create_connection
except ImportError as exc:
    raise SystemExit(
        "websocket-client is required. Install with: pip install websocket-client"
    ) from exc


@dataclass
class Args:
    robot_port: int = 6001
    hostname: str = "127.0.0.1"
    ws_host: str = "127.0.0.1"
    ws_port: int = 18000
    check_only: bool = False
    mock: bool = False
    instruction: Optional[str] = None
    timeout: int = 300
    episode_len: int = 9000
    control_hz: float = 10.0
    show_img: bool = True
    dry_run: bool = False
    crop_top_camera: bool = False
    action_chunk_len: Optional[int] = None
    # Linearly resample each executed action chunk to this many control steps.
    # The server currently returns 16 actions, so 32 makes the motion about 2x slower.
    interpolated_chunk_len: Optional[int] = 32
    temporal_ensemble: bool = False
    ensemble_temperature: float = 0.5
    ensemble_stride_divisor: float = 2.0
    record_video: bool = True
    video_path: Optional[str] = None
    video_dir: str = "outputs/ws_inference_videos"
    video_fps: Optional[float] = None
    # Async mode: inference runs in a background thread while the robot keeps
    # executing the current chunk. Pair with dobot/inference_rtc.sh on the server.
    async_inference: bool = False
    # Send the next request when this many actions are left in the queue.
    # Must be >= the inference latency in control steps, or the queue drains.
    async_trigger: int = 8
    # Denoising steps per request; None = server default.
    num_inference_steps: Optional[int] = None


image_left = None
image_right = None
image_top = None
thread_run = False
image_lock = threading.Lock()


def ws_url(host: str, port: int) -> str:
    return f"ws://{host}:{port}/ws"


# WebSocket protocol copied from dobot_client/client_ws.py:
#   URL: ws://{host}:{port}/ws
#   request fields: type, images, proprio_data, optional instruction
#   response field: predicted_actions
#
# Changes needed for robot execution:
#   1. Keep one persistent WebSocket connection instead of reconnecting per request,
#      because a robot loop should avoid unnecessary connection jitter.
#   2. Accept already-encoded camera frames from memory instead of image file paths.
#   3. Validate predicted_actions as an action chunk with shape [T, 14].
class MotusWebSocketClient:
    def __init__(self, host: str, port: int, timeout: int) -> None:
        self.url = ws_url(host, port)
        self.timeout = timeout
        self.ws = None

    def connect(self) -> None:
        self.ws = create_connection(self.url, timeout=self.timeout)

    def close(self) -> None:
        if self.ws is not None:
            self.ws.close()
            self.ws = None

    def request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self.ws is None:
            self.connect()
        assert self.ws is not None
        self.ws.send(json.dumps(payload))
        return json.loads(self.ws.recv())

    def health(self) -> Dict[str, Any]:
        return self.request({"type": "health"})

    def mock(self) -> Dict[str, Any]:
        return self.request({"type": "mock"})

    def infer(
        self,
        instruction: Optional[str],
        images_b64: List[str],
        state: np.ndarray,
        num_inference_steps: Optional[int] = None,
    ) -> Tuple[np.ndarray, float, float]:
        payload: Dict[str, Any] = {
            "type": "inference",
            "images": images_b64,
            "proprio_data": [state.astype(float).tolist()],
        }
        if instruction:
            payload["instruction"] = instruction
        if num_inference_steps:
            payload["num_inference_steps"] = int(num_inference_steps)

        request_start = time.perf_counter()
        response = self.request(payload)
        round_trip_ms = (time.perf_counter() - request_start) * 1000.0
        inference_ms = float(response.get("processing_time_ms", 0.0))
        if response.get("type") == "error":
            raise RuntimeError(f"WebSocket inference error: {response.get('detail')}")
        if response.get("type") != "inference":
            raise RuntimeError(f"Unexpected WebSocket response type: {response.get('type')}")

        actions = np.asarray(response.get("predicted_actions", []), dtype=np.float64)
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)
        if actions.ndim != 2 or actions.shape[1] != 14:
            raise RuntimeError(f"Invalid action shape from server: {actions.shape}")
        if not np.isfinite(actions).all():
            raise RuntimeError("Server returned non-finite action values")
        communication_ms = max(0.0, round_trip_ms - inference_ms)
        return actions, inference_ms, communication_ms


def run_thread_cam(
    rs_cam: RealSenseCamera,
    which_cam: int,
    crop_top_camera: bool = False,
) -> None:
    global image_left, image_right, image_top, thread_run
    while thread_run:
        if which_cam == 1:
            image, _ = rs_cam.read()
            image = image[:, :, ::-1]
            with image_lock:
                image_left = image
        elif which_cam == 2:
            image, _ = rs_cam.read()
            image = image[:, :, ::-1]
            with image_lock:
                image_right = image
        elif which_cam == 0:
            image_src, _ = rs_cam.read()
            if crop_top_camera:
                image_src = image_src[150:420, 220:480, ::-1]
                image = cv2.resize(image_src, (640, 480))
            else:
                image = image_src[:, :, ::-1]
            with image_lock:
                image_top = image
        else:
            raise ValueError(f"Invalid camera index: {which_cam}")


def get_current_images() -> List[np.ndarray]:
    with image_lock:
        images = [image_top, image_left, image_right]
        if any(image is None for image in images):
            raise RuntimeError("Camera frames are not ready yet")
        return [image.copy() for image in images]


# Different from dobot_client/client_ws.py::_encode_image_to_base64, which reads
# an image file from disk. Here camera frames already live in memory as OpenCV
# numpy arrays, so we first encode the array to JPEG bytes and then base64 it.
# The server still receives the same kind of base64 image string.
def encode_image_b64(image: np.ndarray, quality: int = 85) -> str:
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("Failed to encode camera image")
    return base64.b64encode(encoded.tobytes()).decode("utf-8")


def make_video_path(video_path: Optional[str], video_dir: str) -> str:
    if video_path:
        return video_path
    os.makedirs(video_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    return os.path.join(video_dir, f"ws_inference_{timestamp}.mp4")


def make_video_writer(
    path: str,
    frame: np.ndarray,
    fps: float,
) -> cv2.VideoWriter:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    height, width = frame.shape[:2]
    writer = cv2.VideoWriter(
        path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {path}")
    return writer


def clamp_grippers(action: np.ndarray) -> np.ndarray:
    action = action.copy()
    action[6] = np.clip(action[6], 0.0, 1.0)
    action[13] = np.clip(action[13], 0.0, 1.0)
    return action


def interpolate_action_chunk(
    actions: np.ndarray,
    target_len: Optional[int],
) -> np.ndarray:
    """Linearly upsample an action chunk while preserving both endpoints."""
    if target_len is None or target_len == len(actions):
        return actions.copy()
    if len(actions) == 0:
        raise ValueError("Cannot interpolate an empty action chunk")
    if target_len < len(actions):
        raise ValueError(
            "interpolated_chunk_len only supports slowing down (upsampling): "
            f"{target_len} < {len(actions)}"
        )
    if len(actions) == 1:
        return np.repeat(actions, target_len, axis=0)

    source_steps = np.linspace(0.0, 1.0, len(actions))
    target_steps = np.linspace(0.0, 1.0, target_len)
    return np.column_stack(
        [np.interp(target_steps, source_steps, actions[:, i]) for i in range(actions.shape[1])]
    )


def blend_overlapping_actions(
    old_actions: np.ndarray,
    new_actions: np.ndarray,
    temperature: float,
) -> np.ndarray:
    if old_actions.shape != new_actions.shape:
        raise ValueError(
            f"Cannot blend action chunks with different shapes: "
            f"{old_actions.shape} vs {new_actions.shape}"
        )
    old_weight = np.exp(-temperature)
    new_weight = 1.0
    return (old_actions * old_weight + new_actions * new_weight) / (old_weight + new_weight)


def move_linearly(
    env: RobotEnv,
    start: np.ndarray,
    target: np.ndarray,
    max_steps: int,
    max_step: float = 0.001,
) -> None:
    max_delta = float(np.max(np.abs(start - target)))
    steps = max(1, min(int(max_delta / max_step), max_steps))
    for joint_state in np.linspace(start, target, steps):
        env.step(joint_state, np.array([1, 1]))


def init_cameras(crop_top_camera: bool = False) -> List[threading.Thread]:
    global thread_run
    thread_run = True
    camera_dict = load_ini_data_camera()

    rs_top = RealSenseCamera(flip=True, device_id=camera_dict["top"])
    rs_left = RealSenseCamera(flip=False, device_id=camera_dict["left"])
    rs_right = RealSenseCamera(flip=True, device_id=camera_dict["right"])

    threads = [
        threading.Thread(target=run_thread_cam, args=(rs_top, 0, crop_top_camera), daemon=True),
        threading.Thread(target=run_thread_cam, args=(rs_left, 1), daemon=True),
        threading.Thread(target=run_thread_cam, args=(rs_right, 2), daemon=True),
    ]
    for thread in threads:
        thread.start()

    time.sleep(2)
    get_current_images()
    print("camera thread init success...")
    return threads


def init_robot(args: Args) -> RobotEnv:
    robot_client = ZMQClientRobot(port=args.robot_port, host=args.hostname)
    env = RobotEnv(robot_client)
    if not args.dry_run:
        env.set_do_status([1, 0])
        env.set_do_status([2, 0])
        env.set_do_status([3, 0])
    print("robot init success...")

    if args.dry_run:
        print("dry_run enabled: skip robot init motion")
        return env

    reset_left = np.deg2rad([-90, 30, -110, 20, 90, 90, 0])
    reset_right = np.deg2rad([90, -30, 110, -20, -90, -90, 0])
    reset_joints = np.concatenate([reset_left, reset_right])
    curr_joints = env.get_obs()["joint_positions"]
    move_linearly(env, curr_joints, reset_joints, max_steps=150)
    time.sleep(1)

    photo_left = np.deg2rad([-90, 0, -90, 0, 90, 90, 57])
    photo_right = np.deg2rad([90, 0, 90, 0, -90, -90, 57])
    photo_joints = np.concatenate([photo_left, photo_right])
    curr_joints = env.get_obs()["joint_positions"]
    move_linearly(env, curr_joints, photo_joints, max_steps=150)
    return env


def run_async(
    args: Args,
    env: RobotEnv,
    ws_client: MotusWebSocketClient,
    observation: Dict[str, Any],
    last_action: np.ndarray,
    record_frame,
) -> None:
    """Background inference + continuous execution (async double buffer).

    The main thread executes one action per control tick. When the queue drops to
    async_trigger, it snapshots images + qpos and hands them to the worker. When the
    new chunk arrives, the actions executed since that snapshot are skipped, because
    chunk[0] is the step right after the snapshot state. The server-side RTC guides
    that skipped prefix toward the old plan, so the splice point is continuous.
    """
    interval = 1.0 / args.control_hz
    lock = threading.Lock()
    wake = threading.Event()
    shared: Dict[str, Any] = {"request": None, "result": None, "stop": False}

    def request_chunk(images: List[np.ndarray], qpos: np.ndarray) -> Tuple[np.ndarray, float, float]:
        images_b64 = [encode_image_b64(image) for image in images]
        actions, inference_ms, communication_ms = ws_client.infer(
            args.instruction, images_b64, qpos, args.num_inference_steps
        )
        actions = np.asarray([clamp_grippers(action) for action in actions], dtype=np.float64)
        if args.action_chunk_len is not None:
            actions = actions[: args.action_chunk_len]
        actions = interpolate_action_chunk(actions, args.interpolated_chunk_len)
        return actions, inference_ms, communication_ms

    def worker() -> None:
        while True:
            wake.wait()
            wake.clear()
            with lock:
                if shared["stop"]:
                    return
                req = shared["request"]
            if req is None:
                continue
            try:
                actions, inference_ms, communication_ms = request_chunk(*req)
                result = (actions, inference_ms, communication_ms, None)
            except Exception as exc:  # surfaced in the main thread
                result = (None, 0.0, 0.0, exc)
            with lock:
                shared["result"] = result
                shared["request"] = None

    images = get_current_images()
    record_frame(images)
    actions, inference_ms, communication_ms = request_chunk(images, observation["qpos"].copy())
    queue = deque(actions)
    print(
        f"[async] bootstrap chunk={len(queue)} inference_ms={inference_ms:.1f} "
        f"communication_ms={communication_ms:.1f}"
    )

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    first = True
    t = 0
    exec_count = 0
    pending_mark: Optional[int] = None
    drained_steps = 0
    try:
        while t < args.episode_len:
            step_start = time.time()

            with lock:
                ready = shared["result"]
                shared["result"] = None
            if ready is not None:
                actions, inference_ms, communication_ms, error = ready
                if error is not None:
                    raise RuntimeError(f"Async inference failed: {error}") from error
                executed = exec_count - pending_mark
                skip = min(executed, len(actions) - 1)
                queue = deque(actions[skip:])
                print(
                    f"[async] new chunk inference_ms={inference_ms:.1f} "
                    f"communication_ms={communication_ms:.1f} executed_during_infer={executed} "
                    f"skip={skip} queue={len(queue)} drained_steps={drained_steps}"
                )
                pending_mark = None
                drained_steps = 0

            if pending_mark is None and len(queue) <= args.async_trigger:
                images = get_current_images()
                if args.show_img:
                    cv2.imshow("imgs", np.hstack(images))
                    cv2.waitKey(1)
                with lock:
                    shared["request"] = (images, observation["qpos"].copy())
                pending_mark = exec_count
                wake.set()

            if not queue:
                # Inference slower than the remaining queue: hold the last command.
                drained_steps += 1
            else:
                action = queue.popleft()
                if first and not args.dry_run:
                    move_linearly(env, last_action, action, max_steps=100)
                    first = False
                elif first:
                    first = False

                if args.dry_run:
                    obs = env.get_obs()
                else:
                    obs = env.step(action, np.array([1, 1]))

                if args.record_video:
                    record_frame(get_current_images())

                obs["joint_positions"][6] = action[6]
                obs["joint_positions"][13] = action[13]
                observation["qpos"] = obs["joint_positions"].copy()
                last_action = action.copy()
                t += 1
                exec_count += 1

            elapsed = time.time() - step_start
            if interval > elapsed:
                time.sleep(interval - elapsed)
    finally:
        with lock:
            shared["stop"] = True
        wake.set()
        thread.join(timeout=args.timeout)


def main(args: Args) -> int:
    global thread_run
    if args.action_chunk_len is not None and args.action_chunk_len <= 0:
        raise ValueError("action_chunk_len must be positive when set")
    if args.interpolated_chunk_len is not None and args.interpolated_chunk_len <= 0:
        raise ValueError("interpolated_chunk_len must be positive when set")
    if args.async_inference:
        if args.temporal_ensemble:
            raise ValueError("async_inference and temporal_ensemble both blend chunk seams; enable only one")
        if args.control_hz <= 0:
            raise ValueError("async_inference requires control_hz > 0")
        if args.async_trigger < 1:
            raise ValueError("async_trigger must be >= 1")
    if args.ensemble_temperature <= 0:
        raise ValueError("ensemble_temperature must be positive")
    if args.ensemble_stride_divisor <= 1:
        raise ValueError("ensemble_stride_divisor must be greater than 1")
    if args.video_fps is not None and args.video_fps <= 0:
        raise ValueError("video_fps must be positive when set")

    ws_client = MotusWebSocketClient(args.ws_host, args.ws_port, args.timeout)
    video_writer = None
    resolved_video_path = None

    try:
        # Same standalone WebSocket checks as dobot_client/client_ws.py. These run
        # before camera/robot init so they can safely test server connectivity only.
        if args.check_only:
            health = ws_client.health()
            print(json.dumps(health, indent=2, ensure_ascii=False))
            ok = health.get("type") == "health" and health.get("status") == "healthy"
            print("check-only:", "PASS" if ok else "FAIL")
            return 0 if ok else 1

        if args.mock:
            result = ws_client.mock()
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0 if result.get("type") == "mock" else 1

        init_cameras(crop_top_camera=args.crop_top_camera)
        env = init_robot(args)

        health = ws_client.health()
        print("WebSocket health:", health)
        if health.get("status") != "healthy":
            raise RuntimeError(f"WebSocket server is not healthy: {health}")

        obs = env.get_obs()
        obs["joint_positions"][6] = 1.0
        obs["joint_positions"][13] = 1.0
        observation = {
            "qpos": obs["joint_positions"].copy(),
            "images": {"left_wrist": None, "right_wrist": None, "top": None},
        }
        last_action = observation["qpos"].copy()

        first = True
        t = 0
        interval = 1.0 / args.control_hz if args.control_hz > 0 else 0.0
        if args.record_video:
            resolved_video_path = make_video_path(args.video_path, args.video_dir)
            print(f"Recording camera video to: {resolved_video_path}")
        # Legacy local-model loop reset reference. Not used in the server chunk workflow.
        # initial_action = np.deg2rad([-90, 0, -90, 0, 90, 90, 57, 90, 0, 90, 0, -90, -90, 57])
        prev_action_chunk = None
        if args.temporal_ensemble:
            print(
                "Temporal ensemble enabled:",
                f"temperature={args.ensemble_temperature}",
                f"stride_divisor={args.ensemble_stride_divisor}",
                "mode=chunk_overlap",
            )
        print("The robot begins to perform tasks with WebSocket inference...")

        if args.async_inference:
            def record_frame(images: List[np.ndarray]) -> None:
                nonlocal video_writer
                if not args.record_video:
                    return
                video_frame = np.hstack(images)
                if video_writer is None:
                    assert resolved_video_path is not None
                    video_writer = make_video_writer(
                        resolved_video_path, video_frame, args.video_fps or args.control_hz
                    )
                video_writer.write(video_frame)

            print(f"[async] enabled: trigger={args.async_trigger} control_hz={args.control_hz}")
            run_async(args, env, ws_client, observation, last_action, record_frame)
            return 0

        while t < args.episode_len:
            chunk_start = time.time()
            images = get_current_images()
            observation["images"]["top"] = images[0]
            observation["images"]["left_wrist"] = images[1]
            observation["images"]["right_wrist"] = images[2]

            if args.show_img:
                cv2.imshow("imgs", np.hstack(images))
                cv2.waitKey(1)

            if args.record_video:
                video_frame = np.hstack(images)
                if video_writer is None:
                    assert resolved_video_path is not None
                    video_fps = args.video_fps or args.control_hz
                    video_writer = make_video_writer(resolved_video_path, video_frame, video_fps)
                video_writer.write(video_frame)

            images_b64 = [encode_image_b64(image) for image in images]
            actions, inference_ms, communication_ms = ws_client.infer(
                args.instruction, images_b64, observation["qpos"], args.num_inference_steps
            )
            actions = np.asarray([clamp_grippers(action) for action in actions], dtype=np.float64)
            server_chunk_len = len(actions)
            effective_chunk_len = None
            ensemble_phase = None
            if args.temporal_ensemble:
                effective_chunk_len = args.action_chunk_len or server_chunk_len
                if effective_chunk_len > server_chunk_len:
                    raise ValueError(
                        "action_chunk_len cannot be larger than the returned chunk length "
                        f"in temporal_ensemble mode: {effective_chunk_len} > {server_chunk_len}"
                    )
                actions = actions[:effective_chunk_len]
                stride = max(1, int(effective_chunk_len / args.ensemble_stride_divisor))
                if stride >= effective_chunk_len:
                    raise ValueError(
                        "temporal_ensemble requires action_chunk_len / ensemble_stride_divisor "
                        f"to be smaller than action_chunk_len: {stride} >= {effective_chunk_len}"
                    )

                if prev_action_chunk is None:
                    actions_to_execute = actions[:stride]
                    ensemble_phase = "warmup"
                else:
                    overlap_len = min(stride, len(prev_action_chunk) - stride, len(actions))
                    if overlap_len <= 0:
                        raise RuntimeError(
                            "No overlapping actions available for temporal ensemble: "
                            f"prev_chunk={len(prev_action_chunk)}, current_chunk={len(actions)}, "
                            f"stride={stride}"
                        )
                    old_tail = prev_action_chunk[stride: stride + overlap_len]
                    new_head = actions[:overlap_len]
                    actions_to_execute = blend_overlapping_actions(
                        old_tail,
                        new_head,
                        args.ensemble_temperature,
                    )
                    ensemble_phase = "overlap"
                prev_action_chunk = actions.copy()
                actions = actions_to_execute
            elif args.action_chunk_len is not None:
                actions = actions[: args.action_chunk_len]
            source_chunk_len = len(actions)
            actions = interpolate_action_chunk(actions, args.interpolated_chunk_len)
            print(
                "inference_ms:",
                inference_ms,
                "communication_ms:",
                communication_ms,
                "chunk:",
                len(actions),
                "server_chunk:",
                server_chunk_len,
                "effective_chunk:",
                effective_chunk_len,
                "interpolated:",
                f"{source_chunk_len}->{len(actions)}",
                "ensemble_phase:",
                ensemble_phase,
                "stride:",
                stride if args.temporal_ensemble else None,
            )

            for chunk_idx, action in enumerate(actions):
                if t >= args.episode_len:
                    break

                step_start = time.time()

                delta = action - last_action
                print("Joint increment:", delta)

                if first and not args.dry_run:
                    move_linearly(env, last_action, action, max_steps=100)
                    first = False
                elif first:
                    first = False

                if args.dry_run:
                    obs = env.get_obs()
                else:
                    obs = env.step(action, np.array([1, 1]))

                if args.record_video:
                    step_images = get_current_images()
                    video_frame = np.hstack(step_images)
                    if video_writer is None:
                        assert resolved_video_path is not None
                        video_fps = args.video_fps or args.control_hz
                        video_writer = make_video_writer(resolved_video_path, video_frame, video_fps)
                    video_writer.write(video_frame)

                obs["joint_positions"][6] = action[6]
                obs["joint_positions"][13] = action[13]
                observation["qpos"] = obs["joint_positions"].copy()
                last_action = action.copy()
                t += 1

                # Legacy behavior from run_inference.py: reset t when the robot returns
                # to its initial pose. For server-driven chunks, keep t as a plain local
                # step counter.
                # threshold = np.deg2rad(10)
                # if t > 1200 and np.all(np.abs(action - initial_action) < threshold):
                #     print("Reset t=0")
                #     t = 0

                elapsed = time.time() - step_start
                print("step:", t, "chunk_idx:", chunk_idx, "step ms:", elapsed * 1000)
                if interval > elapsed:
                    time.sleep(interval - elapsed)

            print("chunk total ms:", (time.time() - chunk_start) * 1000)

        return 0

    finally:
        thread_run = False
        if video_writer is not None:
            video_writer.release()
            if resolved_video_path is not None:
                print(f"Saved camera video to: {resolved_video_path}")
        ws_client.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main(tyro.cli(Args)))
