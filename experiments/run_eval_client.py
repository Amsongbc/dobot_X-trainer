#!/usr/bin/env python3
"""Standalone Dobot evaluation client for the openpi_rlinf cobot infer server.

The environment half is taken from ``run_stage2_env_client.py`` (RealSense
capture threads, Dobot reset / step, gripper state override, keyboard verdicts,
fault handling) so the observation distribution stays byte-identical to the one
the policy was trained and collected on.  The transport half is taken from
``openpi_rlinf_cobot_infer_client.py``: plain HTTP, ``GET /health`` plus
``POST /infer`` with base64 JPEG images.

The task instruction lives on the server: the ``prompt`` key is deliberately
left out of every payload so the server applies its own ``default_prompt``
(visible in ``GET /health``).  Note that omitting the key and sending
``"prompt": null`` are NOT equivalent -- the latter reaches the model as the
literal string ``"None"`` -- so never send the key at all.

There is no RL machinery here -- no replay buffer, no human intervention, no
actor switching, no reward shaping.  Evaluation data collection is optional;
when enabled it uses the original ``run_control.py`` directory and file layout.

Usage:

    # No hardware needed: connectivity check with random images (like the
    # reference client's --smoke-test).
    python run_eval_client.py --server-url http://127.0.0.1:8000

    # Real evaluation on the robot:
    python run_eval_client.py --eval --server-url http://127.0.0.1:8000 \
        --chunk-length 10

    # Evaluate and save run_control-style collection data:
    python run_eval_client.py --eval --collect-data

The original direct-HTTP behavior is unchanged.  For the NSCC server, add
``--ssh-tunnel`` to let this client open the required jump-host tunnel and close
it automatically on exit.

Every episode is ended by the operator, never by a step budget: ``s``/space =
success, ``f`` = failure, ``r`` = discard the episode (bad scene setup or
operator error; not counted).  The run keeps handing out episodes until
``Ctrl+C``, which stops and reports the success rate collected so far.

Place this file next to ``run_stage2_env_client.py`` -- it uses the same
``BASE_DIR`` sys.path trick to reach ``dobot_control`` and ``scripts``.
"""
from __future__ import annotations

import base64
import io
import json
import math
import os
import pickle
import re
import select
import socket
import subprocess
import sys
import termios
import threading
import time
import tty
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import requests
import tyro
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

try:  # Hardware deps are only needed for --eval; the smoke test runs anywhere.
    import cv2
    from dobot_control.cameras.realsense_camera import RealSenseCamera
    from dobot_control.env import RobotEnv
    from dobot_control.robots.robot_node import ZMQClientRobot
    from scripts.manipulate_utils import load_ini_data_camera

    HARDWARE_IMPORT_ERROR: Optional[BaseException] = None
except Exception as exc:  # pragma: no cover - depends on the robot host
    HARDWARE_IMPORT_ERROR = exc

STATE_DIM = 14
ACTION_DIM = 14
GRIPPER_INDICES = (6, 13)


# --- Dobot hardware helpers (shared with run_stage2_env_client.py) ---
# Frames are published as RGB, matching what the model was trained on.

image_left = None
image_right = None
image_top = None
thread_run = False
image_lock = threading.Lock()


def run_thread_cam(
    rs_cam: "RealSenseCamera",
    which_cam: int,
    crop_top_camera: bool = False,
) -> None:
    """Publish frames as RGB.

    ``RealSenseCamera.read`` already returns RGB: it opens the stream as
    ``rs.format.bgr8`` and reverses the channels itself.  The data-collection
    script reverses them a second time only to feed ``cv2.imwrite``, which wants
    BGR; the LeRobot training videos therefore hold RGB.  Inference clients that
    copied that second reversal without the matching imwrite were sending BGR to
    an RGB-trained model, so it is deliberately absent here.
    """
    global image_left, image_right, image_top
    while thread_run:
        if which_cam == 1:
            image, _ = rs_cam.read()
            with image_lock:
                image_left = image
        elif which_cam == 2:
            image, _ = rs_cam.read()
            with image_lock:
                image_right = image
        elif which_cam == 0:
            image_src, _ = rs_cam.read()
            if crop_top_camera:
                image_src = image_src[150:420, 220:480]
                image = cv2.resize(image_src, (640, 480))
            else:
                image = image_src
            with image_lock:
                image_top = image
        else:
            raise ValueError(f"Invalid camera index: {which_cam}")


def get_current_images() -> list[np.ndarray]:
    with image_lock:
        images = [image_top, image_left, image_right]
        if any(image is None for image in images):
            raise RuntimeError("Camera frames are not ready yet")
        return [image.copy() for image in images]


def init_cameras(crop_top_camera: bool = False) -> list[threading.Thread]:
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


def stop_cameras() -> None:
    global thread_run
    thread_run = False


def move_linearly(
    env: "RobotEnv",
    start: np.ndarray,
    target: np.ndarray,
    max_steps: int,
    max_step: float = 0.001,
) -> None:
    max_delta = float(np.max(np.abs(start - target)))
    steps = max(1, min(int(max_delta / max_step), max_steps))
    for joint_state in np.linspace(start, target, steps):
        env.step(joint_state, np.array([1, 1]))


# --- Keyboard verdicts (trimmed to what an eval run needs) ---


class KeyboardVerdict:
    """Latch a terminal verdict typed while the episode is running."""

    def __init__(self) -> None:
        self._old_settings = None
        self._raw = False
        self._verdict: Optional[str] = None

    def start(self) -> None:
        self._verdict = None
        try:
            self._old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
            self._raw = True
        except (termios.error, OSError):
            self._raw = False

    def stop(self) -> None:
        if self._raw and self._old_settings is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)
        self._raw = False
        self._old_settings = None

    def check(self) -> Optional[str]:
        if self._verdict is not None:
            return self._verdict
        if self._raw and select.select([sys.stdin], [], [], 0)[0]:
            key = sys.stdin.read(1).lower()
            if key == "\x03":
                raise KeyboardInterrupt
            if key in ("s", " "):
                self._verdict = "success"
            elif key == "f":
                self._verdict = "failure"
            elif key == "r":
                self._verdict = "discarded"
        return self._verdict


def ask_verdict(prompt: str, default: str) -> str:
    """Fall back to an explicit prompt when no key was pressed."""
    labels = {"success": "s=成功", "failure": "f=失败", "discarded": "r=作废"}
    hint = "，".join(labels.values())
    while True:
        choice = input(f"{prompt}（{hint}，直接 Enter={labels[default]}）：").strip().lower()
        if not choice:
            return default
        if choice in ("s", " "):
            return "success"
        if choice == "f":
            return "failure"
        if choice == "r":
            return "discarded"


# --- openpi_rlinf HTTP inference protocol (from the reference infer client) ---


def encode_jpeg_b64(image: np.ndarray, quality: int) -> str:
    """Encode an HWC RGB uint8 frame the way the reference client does."""
    array = np.ascontiguousarray(image, dtype=np.uint8)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"Expected an HWC RGB image, got {array.shape}")
    buffer = io.BytesIO()
    Image.fromarray(array, mode="RGB").save(buffer, format="JPEG", quality=quality)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def encode_random_image(height: int, width: int, quality: int) -> str:
    rng = np.random.default_rng(0)
    return encode_jpeg_b64(
        rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8), quality
    )


def normalize_server_url(url: str) -> str:
    """Accept a bare ``host:port`` too -- requests needs an explicit scheme.

    A regex rather than urlparse: ``urlparse("localhost:8000")`` reads
    ``localhost`` as the scheme, which silently produces a broken URL.
    """
    url = url.strip()
    if not re.match(r"^https?://", url, re.IGNORECASE):
        url = "http://" + url
    return url.rstrip("/")


def prompt_directory_name(prompt: Optional[str]) -> str:
    """Convert the server task prompt to a safe, readable directory name."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("开启数采时，server 必须返回非空 task prompt")
    name = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", prompt.strip())
    name = re.sub(r"\s+", " ", name).strip(" .")
    if not name:
        raise ValueError("server task prompt 无法用作数据目录名")
    # Stay comfortably below common per-component filesystem byte limits while
    # preserving UTF-8 characters and the human-readable task name.
    encoded = name.encode("utf-8")
    if len(encoded) > 180:
        name = encoded[:180].decode("utf-8", errors="ignore").rstrip(" .")
    return name


class InferClient:
    """Thin wrapper over ``GET /health`` and ``POST /infer``."""

    def __init__(self, server_url: str, timeout: float, jpeg_quality: int) -> None:
        base = normalize_server_url(server_url)
        self.health_url = f"{base}/health"
        self.infer_url = f"{base}/infer"
        self.timeout = timeout
        self.jpeg_quality = jpeg_quality
        # Keep-alive: an eval run issues hundreds of /infer calls.
        self.session = requests.Session()
        self.action_dim: Optional[int] = None
        self.chunk_length: Optional[int] = None
        # Whatever prompt the server reports having used, for the report.
        self.server_prompt: Optional[str] = None

    def health(self) -> dict[str, Any]:
        response = self.session.get(self.health_url, timeout=10)
        response.raise_for_status()
        return response.json()

    def build_payload(
        self, state: np.ndarray, images: list[np.ndarray]
    ) -> dict[str, Any]:
        """No ``prompt`` key: the server supplies the task instruction."""
        flat = np.asarray(state, dtype=np.float64).reshape(-1)
        if flat.shape != (STATE_DIM,):
            raise ValueError(f"state must have {STATE_DIM} dims, got {flat.shape}")
        return {
            "state": [float(value) for value in flat],
            "cam_high_b64": encode_jpeg_b64(images[0], self.jpeg_quality),
            "cam_left_wrist_b64": encode_jpeg_b64(images[1], self.jpeg_quality),
            "cam_right_wrist_b64": encode_jpeg_b64(images[2], self.jpeg_quality),
        }

    def post(self, payload: dict[str, Any]) -> tuple[np.ndarray, float]:
        started = time.monotonic()
        response = self.session.post(self.infer_url, json=payload, timeout=self.timeout)
        latency = time.monotonic() - started
        if response.status_code != 200:
            raise RuntimeError(
                f"/infer returned {response.status_code}: {response.text[:500]}"
            )
        data = response.json()
        actions = np.asarray(data["actions"], dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
            raise ValueError(f"Expected actions [C,{ACTION_DIM}], got {actions.shape}")
        self.action_dim = int(data.get("action_dim", actions.shape[1]))
        self.chunk_length = int(data.get("num_action_chunks", actions.shape[0]))
        self.server_prompt = data.get("prompt")
        return actions, latency

    def infer(
        self, state: np.ndarray, images: list[np.ndarray]
    ) -> tuple[np.ndarray, float]:
        return self.post(self.build_payload(state, images))


class SSHTunnel:
    """Own an SSH local-forwarding process for the lifetime of the eval."""

    def __init__(
        self,
        *,
        target: str,
        jump: str,
        local_port: int,
        remote_port: int,
        connect_timeout: float,
    ) -> None:
        self.target = target
        self.jump = jump
        self.local_port = local_port
        self.remote_port = remote_port
        self.connect_timeout = connect_timeout
        self.process: Optional[subprocess.Popen[bytes]] = None

    def start(self) -> None:
        if not 1 <= self.local_port <= 65535:
            raise ValueError(f"Invalid SSH local port: {self.local_port}")
        if not 1 <= self.remote_port <= 65535:
            raise ValueError(f"Invalid SSH remote port: {self.remote_port}")

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.settimeout(0.2)
            if probe.connect_ex(("127.0.0.1", self.local_port)) == 0:
                raise RuntimeError(
                    f"Local port {self.local_port} is already in use; stop the "
                    "old tunnel or choose --ssh-local-port"
                )

        command = [
            "ssh",
            "-N",
            "-o",
            "ExitOnForwardFailure=yes",
            "-o",
            "ServerAliveInterval=30",
            "-o",
            "ServerAliveCountMax=3",
            "-L",
            f"127.0.0.1:{self.local_port}:127.0.0.1:{self.remote_port}",
            "-J",
            self.jump,
            self.target,
        ]
        print(
            f"Opening SSH tunnel: 127.0.0.1:{self.local_port} -> "
            f"{self.target}:127.0.0.1:{self.remote_port}"
        )
        self.process = subprocess.Popen(command)

        deadline = time.monotonic() + self.connect_timeout
        while time.monotonic() < deadline:
            return_code = self.process.poll()
            if return_code is not None:
                self.process = None
                raise RuntimeError(f"SSH tunnel exited with code {return_code}")
            try:
                with socket.create_connection(
                    ("127.0.0.1", self.local_port), timeout=0.5
                ):
                    print("SSH tunnel ready")
                    return
            except OSError:
                time.sleep(0.2)
        self.stop()
        raise TimeoutError(
            f"SSH tunnel did not open within {self.connect_timeout:.1f}s"
        )

    def stop(self) -> None:
        process = self.process
        self.process = None
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


@dataclass
class Args:
    # Run the real robot evaluation.  Without it this is a server smoke test.
    eval: bool = False
    server_url: str = "http://127.0.0.1:8000"
    # Optional NSCC jump-host tunnel. False preserves the original direct-HTTP
    # behavior; enable it explicitly with --ssh-tunnel.
    ssh_tunnel: bool = False
    ssh_target: str = "sysu_xdliang_2@pytorch-ng-1984438-lfwj"
    ssh_jump: str = "e85f315bde7748bab59f40bcf30642f5@proxy.nscc-gz.cn:8022"
    ssh_local_port: int = 18000
    ssh_remote_port: int = 8000
    ssh_connect_timeout: float = 30.0
    # Number of actions consumed from each server chunk. -1 = the whole chunk.
    chunk_length: int = -1
    # Deprecated compatibility option: 0 = the whole chunk. Used only when
    # chunk_length remains -1.
    exec_horizon: int = 0
    # Robot action execution frequency.
    control_hz: float = 10.0
    # ZMQ port used by the robot controller.
    robot_port: int = 6001
    # Hostname used by the robot controller.
    hostname: str = "127.0.0.1"
    # Apply the configured crop to the top-camera image.
    crop_top_camera: bool = False
    # Run environment actions without sending commands to the physical robot.
    dry_run: bool = False
    # HTTP inference request timeout in seconds.
    timeout: float = 300.0
    # Save evaluation trajectories in the original run_control.py layout.
    collect_data: bool = False
    # Dataset root used by run_control.py.
    save_data_path: str = str(Path(__file__).parent.parent.parent / "datasets")
    # 75 reproduces the reference client (PIL default); 95 stays closer to the
    # raw camera frames the policy was trained on.
    jpeg_quality: int = 95
    out: Optional[str] = None


@dataclass
class EpisodeResult:
    index: int
    verdict: str  # success | failure | discarded
    steps: int
    infer_calls: int
    duration: float
    infer_latency_mean: float
    robot_fault: bool
    note: str = ""


def execution_horizon(args: Args, available_actions: int) -> int:
    """Return how many actions to consume from the current server chunk."""
    configured = args.chunk_length
    if configured == -1:
        configured = args.exec_horizon
    if configured <= 0:
        return available_actions
    return min(configured, available_actions)


class DobotEvalClient:
    def __init__(self, args: Args) -> None:
        if HARDWARE_IMPORT_ERROR is not None:
            raise RuntimeError(
                "--eval needs the robot host dependencies, but importing them "
                f"failed: {HARDWARE_IMPORT_ERROR!r}"
            )
        self.args = args
        self.verdict = KeyboardVerdict()
        self.last_action: Optional[np.ndarray] = None
        self.robot_faulted = False
        self.collection_dir: Optional[Path] = None
        self.collection_step = 0
        self.collection_prompt: Optional[str] = None
        init_cameras(crop_top_camera=args.crop_top_camera)
        robot_client = ZMQClientRobot(port=args.robot_port, host=args.hostname)
        self.env = RobotEnv(robot_client)
        if not args.dry_run:
            for channel in (1, 2, 3):
                self.env.set_do_status([channel, 0])

    def close(self) -> None:
        self.verdict.stop()

    # --- environment ---

    def reset_robot(self) -> None:
        if self.args.dry_run:
            return
        reset_left = np.deg2rad([-90, 30, -110, 20, 90, 90, 0])
        reset_right = np.deg2rad([90, -30, 110, -20, -90, -90, 0])
        reset_joints = np.concatenate([reset_left, reset_right])
        current = self.env.get_obs()["joint_positions"]
        move_linearly(self.env, current, reset_joints, max_steps=150)
        time.sleep(1)
        photo_left = np.deg2rad([-90, 0, -90, 0, 90, 90, 57])
        photo_right = np.deg2rad([90, 0, 90, 0, -90, -90, 57])
        photo_joints = np.concatenate([photo_left, photo_right])
        current = self.env.get_obs()["joint_positions"]
        move_linearly(self.env, current, photo_joints, max_steps=150)

    def read_observation(self) -> tuple[np.ndarray, list[np.ndarray]]:
        """State plus the three HWC RGB frames, with the gripper override."""
        images = get_current_images()
        state = np.asarray(self.env.get_obs()["joint_positions"], dtype=np.float32).copy()
        if self.last_action is not None:
            # Commanded gripper, not measured -- identical to the collection client.
            for index in GRIPPER_INDICES:
                state[index] = self.last_action[index]
        return state, images

    def prepare_action(self, proposed: np.ndarray) -> np.ndarray:
        action = np.asarray(proposed, dtype=np.float64).reshape(-1).copy()
        if action.shape != (ACTION_DIM,) or not np.isfinite(action).all():
            raise ValueError(
                f"Invalid action: shape={action.shape}, finite={np.isfinite(action).all()}"
            )
        for index in GRIPPER_INDICES:
            action[index] = np.clip(action[index], 0.0, 1.0)
        return action

    def apply_action(self, proposed: np.ndarray) -> None:
        action = self.prepare_action(proposed)
        if not self.args.dry_run:
            self.env.step(action, np.array([1, 1]))
        self.last_action = action.copy()

    def start_episode_collection(self) -> None:
        """Create one run_control-style timestamp directory for an episode."""
        if not self.args.collect_data:
            self.collection_dir = None
            self.collection_step = 0
            return

        root = Path(self.args.save_data_path).expanduser()
        dataset_name = prompt_directory_name(self.collection_prompt)
        timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
        episode_dir = root / dataset_name / "collect_data" / timestamp
        # Scene setup normally makes timestamps unique. Avoid overwriting if two
        # episodes nevertheless begin in the same second.
        suffix = 1
        while episode_dir.exists():
            episode_dir = (
                root
                / dataset_name
                / "collect_data"
                / "{}_{}".format(timestamp, suffix)
            )
            suffix += 1
        for subdir in ("topImg", "leftImg", "rightImg", "observation"):
            (episode_dir / subdir).mkdir(parents=True, exist_ok=True)
        self.collection_dir = episode_dir
        self.collection_step = 0
        print("本回合数采目录：{}".format(episode_dir))

    def save_collection_step(self, action: np.ndarray) -> None:
        """Save one pre-action sample in run_control.py's on-disk format."""
        if self.collection_dir is None:
            return

        images = get_current_images()
        obs = dict(self.env.get_obs())
        joints = np.asarray(obs["joint_positions"]).copy()
        if self.last_action is not None:
            for index in GRIPPER_INDICES:
                joints[index] = self.last_action[index]
        obs["joint_positions"] = joints
        obs["control"] = np.asarray(action).copy()

        index = self.collection_step
        names = ("topImg", "leftImg", "rightImg")
        for name, image in zip(names, images):
            # Inference frames are RGB; run_control.py writes BGR arrays through
            # cv2.imwrite, so reverse channels to produce the same JPEG colors.
            output = self.collection_dir / name / "{}.jpg".format(index)
            if not cv2.imwrite(str(output), image[:, :, ::-1]):
                raise OSError("保存图像失败: {}".format(output))

        observation_file = (
            self.collection_dir / "observation" / "{}.pkl".format(index)
        )
        with observation_file.open("wb") as handle:
            pickle.dump(obs, handle)
        self.collection_step += 1

    def prepare_episode(self, index: int) -> None:
        self.verdict.stop()
        while True:
            if self.robot_faulted:
                input("机械臂已安全关停。请解除错误、取下夹持物并清空复位路径，按 Enter 尝试复位：")
            else:
                input("请确认夹爪无危险物体且复位路径已清空，按 Enter 开始复位：")
            try:
                self.reset_robot()
                self.robot_faulted = False
                break
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                self.robot_faulted = True
                print(f"复位失败：{exc}")
        input(f"请布置第 {index} 回合场景，确认安全后按 Enter 开始评测：")
        self.start_episode_collection()
        obs = self.env.get_obs()
        self.last_action = np.asarray(obs["joint_positions"], dtype=np.float64).copy()
        for gripper_index in GRIPPER_INDICES:
            self.last_action[gripper_index] = 1.0
        self.verdict.start()
        print(
            f"第 {index} 回合开始：s/空格=成功，f=失败，r=作废本回合，Ctrl+C=停止评测"
        )

    def handle_fault(self, error: BaseException, stage: str) -> str:
        self.verdict.stop()
        self.robot_faulted = True
        print(f"\n机械臂故障（{stage}）：{type(error).__name__}: {error}")
        return ask_verdict("请判定本回合", default="discarded")

    # --- one episode ---

    def run_episode(self, index: int, infer_client: InferClient) -> EpisodeResult:
        self.prepare_episode(index)
        interval = 1.0 / self.args.control_hz
        latencies: list[float] = []
        steps = 0
        result: Optional[str] = None
        note = ""
        fault = False
        started = time.monotonic()

        # The operator ends the episode; there is no step budget.
        while result is None:
            try:
                state, images = self.read_observation()
            except Exception as exc:
                fault = True
                result = self.handle_fault(exc, "读取观测")
                break

            try:
                actions, latency = infer_client.infer(state, images)
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                self.verdict.stop()
                print(f"\n推理请求失败：{type(exc).__name__}: {exc}")
                note = f"infer_error: {exc}"
                result = ask_verdict("请判定本回合", default="discarded")
                break
            latencies.append(latency)

            horizon = execution_horizon(self.args, len(actions))

            # The arm holds its last commanded pose across the inference gap, so
            # restart the pacing clock once the chunk is actually in hand.
            deadline = time.monotonic()
            for action in actions[:horizon]:
                try:
                    prepared_action = self.prepare_action(action)
                except Exception as exc:
                    fault = True
                    result = self.handle_fault(exc, "检查动作")
                    break
                if self.args.collect_data:
                    try:
                        self.save_collection_step(prepared_action)
                    except Exception as exc:
                        self.verdict.stop()
                        print(
                            "\n保存数采失败：{}: {}".format(
                                type(exc).__name__, exc
                            )
                        )
                        note = "collection_error: {}".format(exc)
                        result = ask_verdict("请判定本回合", default="discarded")
                        break
                try:
                    self.apply_action(prepared_action)
                except Exception as exc:
                    fault = True
                    result = self.handle_fault(exc, "执行动作")
                    break
                steps += 1

                signal = self.verdict.check()
                if signal is not None:
                    result = signal
                    break

                deadline += interval
                time.sleep(max(0.0, deadline - time.monotonic()))
                if time.monotonic() - deadline > interval:
                    deadline = time.monotonic()

        self.verdict.stop()
        duration = time.monotonic() - started
        labels = {"success": "成功", "failure": "失败", "discarded": "作废"}
        print(
            f"第 {index} 回合 -> {labels[result]}"
            f"（steps={steps}, infer={len(latencies)}, "
            f"用时 {duration:.1f}s, 平均推理 {1000 * float(np.mean(latencies)) if latencies else 0.0:.0f}ms）\n"
        )
        if self.collection_dir is not None:
            print(
                "本回合已保存 {} 步数采：{}\n".format(
                    self.collection_step, self.collection_dir
                )
            )
        return EpisodeResult(
            index=index,
            verdict=result,
            steps=steps,
            infer_calls=len(latencies),
            duration=duration,
            infer_latency_mean=float(np.mean(latencies)) if latencies else 0.0,
            robot_fault=fault,
            note=note,
        )


# --- reporting ---


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    spread = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, center - spread), min(1.0, center + spread)


def summarize(
    args: Args, results: list[EpisodeResult], server_prompt: Optional[str]
) -> dict[str, Any]:
    success = sum(1 for r in results if r.verdict == "success")
    failure = sum(1 for r in results if r.verdict == "failure")
    discarded = sum(1 for r in results if r.verdict == "discarded")
    counted = success + failure
    rate = success / counted if counted else 0.0
    low, high = wilson_interval(success, counted)
    latencies = [r.infer_latency_mean for r in results if r.infer_calls]

    print("\n=== 评测结果 ===")
    print(f"server      : {normalize_server_url(args.server_url)}")
    print(f"server task : {server_prompt!r}")
    print(f"成功率      : {success}/{counted} = {100 * rate:.1f}%"
          f"（95% Wilson CI {100 * low:.1f}% - {100 * high:.1f}%）")
    print(f"失败        : {failure}")
    print(f"作废(不计入): {discarded}"
          f"，其中机械臂故障 {sum(1 for r in results if r.robot_fault)}")
    if counted:
        succeeded = [r.steps for r in results if r.verdict == "success"]
        if succeeded:
            print(f"成功回合步数: 均值 {np.mean(succeeded):.1f}，中位 {np.median(succeeded):.0f}")
    if latencies:
        print(f"推理延迟    : 均值 {1000 * float(np.mean(latencies)):.0f}ms")

    return {
        "server_url": normalize_server_url(args.server_url),
        "server_prompt": server_prompt,
        "control_hz": args.control_hz,
        "chunk_length": args.chunk_length,
        "exec_horizon": args.exec_horizon,
        "collect_data": args.collect_data,
        "save_data_path": str(Path(args.save_data_path).expanduser()),
        "dataset_name": (
            prompt_directory_name(server_prompt)
            if args.collect_data and server_prompt
            else None
        ),
        "jpeg_quality": args.jpeg_quality,
        "success": success,
        "failure": failure,
        "discarded": discarded,
        "counted": counted,
        "success_rate": rate,
        "wilson_95": [low, high],
        "episodes": [asdict(r) for r in results],
    }


# --- entry points ---


def run_smoke_test(args: Args) -> int:
    """Server connectivity check with random images -- no robot required."""
    client = InferClient(args.server_url, args.timeout, args.jpeg_quality)
    print(f"Health check: {client.health_url}")
    print(f"  -> {client.health()}")

    height, width = 480, 640
    payload = {
        "state": [0.0] * STATE_DIM,
        "cam_high_b64": encode_random_image(height, width, args.jpeg_quality),
        "cam_left_wrist_b64": encode_random_image(height, width, args.jpeg_quality),
        "cam_right_wrist_b64": encode_random_image(height, width, args.jpeg_quality),
    }
    print(f"Smoke test: random {height}x{width} images -> {client.infer_url}")
    actions, latency = client.post(payload)
    print(
        f"OK shape=({client.chunk_length}, {client.action_dim}) "
        f"latency={1000 * latency:.0f}ms, server prompt={client.server_prompt!r}"
    )
    preview = ", ".join(f"{v:.4f}" for v in actions[0][:6])
    print(f"  first step: [{preview}, ...]")
    print("\n服务器可用。加 --eval 在真机上跑成功率评测。")
    return 0


def run_eval(args: Args) -> int:
    if args.control_hz <= 0:
        raise ValueError("control_hz must be positive")

    if HARDWARE_IMPORT_ERROR is not None:
        print(
            "--eval needs the robot host dependencies (cv2 / dobot_control / "
            f"scripts), but importing them failed: {HARDWARE_IMPORT_ERROR!r}",
            file=sys.stderr,
        )
        return 1

    infer_client = InferClient(args.server_url, args.timeout, args.jpeg_quality)
    print(f"Health check: {infer_client.health_url}")
    health = infer_client.health()
    print(f"  -> {health}")

    client: Optional[DobotEvalClient] = None
    results: list[EpisodeResult] = []
    try:
        client = DobotEvalClient(args)

        # Warm up the server once so compilation latency does not land inside
        # the first episode and stall the arm mid-motion.
        state, images = client.read_observation()
        _, latency = infer_client.infer(state, images)
        print(
            f"warmup infer ok: chunk=({infer_client.chunk_length}, "
            f"{infer_client.action_dim}), latency={1000 * latency:.0f}ms"
        )
        if not infer_client.server_prompt:
            infer_client.server_prompt = health.get("default_prompt")
        client.collection_prompt = infer_client.server_prompt
        if args.collect_data:
            dataset_name = prompt_directory_name(client.collection_prompt)
            print(
                "数采任务目录：{}".format(
                    Path(args.save_data_path).expanduser() / dataset_name
                )
            )
        print(f"任务由 server 指定：{infer_client.server_prompt!r}")

        print("开始评测，Ctrl+C 结束并汇总成功率\n")
        index = 0
        while True:
            index += 1
            results.append(client.run_episode(index, infer_client))
    except KeyboardInterrupt:
        print("\n评测被中断，汇总已完成的回合")
    finally:
        if client is not None:
            client.close()
        stop_cameras()
        if HARDWARE_IMPORT_ERROR is None:
            cv2.destroyAllWindows()

    if not results:
        print("没有完成任何回合")
        return 1

    report = summarize(args, results, infer_client.server_prompt)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
        print(f"结果已写入 {args.out}")
    return 0


def main(args: Args) -> int:
    if args.chunk_length == 0 or args.chunk_length < -1:
        raise ValueError("chunk_length must be -1 or a positive integer")

    tunnel: Optional[SSHTunnel] = None
    if args.ssh_tunnel:
        args.server_url = f"http://127.0.0.1:{args.ssh_local_port}"
        tunnel = SSHTunnel(
            target=args.ssh_target,
            jump=args.ssh_jump,
            local_port=args.ssh_local_port,
            remote_port=args.ssh_remote_port,
            connect_timeout=args.ssh_connect_timeout,
        )
    try:
        if tunnel is not None:
            tunnel.start()
        if not args.eval:
            return run_smoke_test(args)
        return run_eval(args)
    finally:
        if tunnel is not None:
            tunnel.stop()


if __name__ == "__main__":
    raise SystemExit(main(tyro.cli(Args)))
