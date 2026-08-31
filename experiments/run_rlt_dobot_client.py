#!/usr/bin/env python3
"""RLT online-RL Dobot client with chunk rewind and leader takeover.

Port of ``franka_inference_node_rlt.py`` onto the Dobot bimanual stack.  It
replaces the old ``run_stage2_env_client.py`` flow entirely:

* Transport is the RLT websocket protocol (``rlt-online-rl/v1``) spoken through
  ``openpi_client.WebsocketClientPolicy``, not the msgpack RPC env server.  The
  client drives the episode and calls ``act`` / ``transition`` / ``episode_end``;
  the old design had the server drive the client.
* The single-point ``a`` anchor is gone.  Rewind is now per action chunk and
  happens mid-episode: ``b`` arms rewind mode, ``r`` reverse-plays one finished
  chunk, ``q`` marks a chunk bad without moving, and leaving rewind reports the
  correction to the server so it can re-credit the replay buffer.
* Human control is a whole-body leader takeover toggled with ``i`` (franka's
  Gello key), not the per-arm recording-button intervention.

Keyboard (identical to the franka node; conflicting Dobot keys were changed):
    s or Space : success, reward=+1.0, end episode
    f          : failure, reward=0.0, end episode
    p          : progress, reward=+0.5, continue
    o          : small progress, +0.1 per press (stacks), continue
    x          : regress, reward=-0.5, continue
    n          : after an episode ends, start the next one
    b          : enter/cancel rewind mode after the current chunk finishes
    r          : while rewind mode is active, reverse-play one finished chunk
    q          : mark one previous replay chunk as bad without moving the robot
    i          : request/cancel leader takeover after the current chunk finishes
    Ctrl+C     : stop

Franka keeps the rewind pose history in its ROS control node because that node
owns the EE stream.  The Dobot client commands joint targets directly, so the
history lives here and a rewind is a straight reverse replay of the joint
commands that were sent -- no IK and no pose estimation involved.

Usage (tunnel handled in-process, no second terminal):

    python run_rlt_dobot_client.py \
        --ssh-host sysu_xdliang_2@pytorch-ng-1984438-lfwj \
        --ssh-jump e85f...@proxy.nscc-gz.cn:8022 \
        --instruction "cook vegetable"

Place next to run_stage2_env_client.py -- it uses the same BASE_DIR sys.path
trick to reach dobot_control and scripts.
"""
from __future__ import annotations

import os
import re
import select
import shutil
import socket
import subprocess
import sys
import tempfile
import termios
import threading
import time
import tty
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import numpy as np
import tyro
from openpi_client import websocket_client_policy as _websocket_client_policy

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)
from dobot_control.agents.agent import BimanualAgent
from dobot_control.agents.dobot_agent import DobotAgent
from dobot_control.cameras.realsense_camera import RealSenseCamera
from dobot_control.env import RobotEnv
from dobot_control.robots.robot_node import ZMQClientRobot
from scripts.manipulate_utils import load_ini_data_camera, load_ini_data_hands

ACTION_DIM = 14
GRIPPER_INDICES = (6, 13)
RLT_PROTOCOL = "rlt-online-rl/v1"


# --- Dobot hardware helpers (shared with run_stage2_env_client.py) ---
# Frames are published as RGB, matching what the model was trained on.

image_left = None
image_right = None
image_top = None
thread_run = False
image_lock = threading.Lock()


def run_thread_cam(
    rs_cam: RealSenseCamera,
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


def make_leader_agent() -> BimanualAgent:
    _, hands = load_ini_data_hands()
    return BimanualAgent(
        DobotAgent(which_hand="LEFT", dobot_config=hands["HAND_LEFT"]),
        DobotAgent(which_hand="RIGHT", dobot_config=hands["HAND_RIGHT"]),
    )


# --- SSH tunnel (so the client needs no second terminal) ---


def normalize_ssh_target(value: str) -> str:
    return value.strip()


class SSHTunnel:
    """Own an ``ssh -N -L`` child process for the lifetime of the client.

    The system ssh binary is driven on purpose rather than a pure-Python SSH
    library: ssh_config aliases, ProxyJump, agent keys and known_hosts all keep
    working unchanged.  ``-f`` is deliberately NOT passed -- a forked ssh
    detaches from this process and could no longer be shut down or restarted.
    """

    def __init__(
        self,
        host: str,
        local_port: int,
        remote_port: int,
        jump: Optional[str] = None,
        remote_host: str = "127.0.0.1",
        connect_timeout: float = 30.0,
    ) -> None:
        self.host = host
        self.local_port = local_port
        self.remote_port = remote_port
        self.jump = jump
        self.remote_host = remote_host
        self.connect_timeout = connect_timeout
        self._process: Optional[subprocess.Popen] = None
        self._log = None
        self._adopted = False

    @property
    def command(self) -> list[str]:
        command = [
            "ssh",
            "-N",
            "-o", "ExitOnForwardFailure=yes",
            "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=3",
            # Key auth only: an interactive prompt would hang a headless run.
            "-o", "BatchMode=yes",
            "-L",
            f"127.0.0.1:{self.local_port}:{self.remote_host}:{self.remote_port}",
        ]
        if self.jump:
            command += ["-J", self.jump]
        command.append(self.host)
        return command

    def _port_taken(self) -> bool:
        """Probe by binding, not by connecting, so no stray websocket appears."""
        probe = socket.socket()
        try:
            probe.bind(("127.0.0.1", self.local_port))
        except OSError:
            return True
        finally:
            probe.close()
        return False

    def _read_log(self) -> str:
        if self._log is None:
            return ""
        try:
            self._log.seek(0)
            return self._log.read().strip() or "(ssh 没有输出)"
        except (OSError, ValueError):
            return ""

    def _close_log(self) -> None:
        if self._log is not None:
            try:
                self._log.close()
            except OSError:
                pass
            self._log = None

    def ensure(self) -> None:
        if self._adopted:
            return
        if self._process is not None and self._process.poll() is None:
            return
        if self._process is not None:
            print(f"SSH 隧道已断开（ssh 退出码 {self._process.returncode}），正在重建 ...")
        elif self._port_taken():
            print(f"检测到 127.0.0.1:{self.local_port} 已被占用，复用现有转发")
            self._adopted = True
            return
        self._start()

    def _start(self) -> None:
        if shutil.which("ssh") is None:
            raise RuntimeError("找不到 ssh 可执行文件")
        self._close_log()
        self._log = tempfile.TemporaryFile(mode="w+")
        print("启动 SSH 隧道：" + " ".join(self.command))
        self._process = subprocess.Popen(
            self.command,
            stdin=subprocess.DEVNULL,
            stdout=self._log,
            stderr=subprocess.STDOUT,
        )
        deadline = time.monotonic() + self.connect_timeout
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                message = self._read_log()
                self._close_log()
                raise RuntimeError(f"SSH 隧道启动失败：{message}")
            if self._port_taken():
                print(
                    f"SSH 隧道就绪：127.0.0.1:{self.local_port} -> "
                    f"{self.host} 的 {self.remote_host}:{self.remote_port}"
                )
                return
            time.sleep(0.3)
        message = self._read_log()
        self.close()
        raise RuntimeError(f"SSH 隧道 {self.connect_timeout:.0f}s 内未就绪：{message}")

    def close(self) -> None:
        process, self._process = self._process, None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
            print("SSH 隧道已关闭")
        self._close_log()


# --- Keyboard ---


class KeyboardControl:
    """Non-blocking keyboard reader using the franka node's key assignment."""

    KEYS = {
        "s": "success",
        " ": "success",
        "f": "failure",
        "p": "progress",
        "o": "progress_small",
        "x": "regress",
        "n": "next_episode",
        "b": "rewind_mode",
        "r": "rewind_step",
        "q": "rewind_credit",
        "i": "takeover",
    }

    def __init__(self) -> None:
        self._old_settings = None
        self._raw = False

    def start(self) -> None:
        if self._raw:
            return
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

    def poll(self) -> list[str]:
        """Drain every buffered keypress, so fast typing is never dropped."""
        events: list[str] = []
        if not self._raw:
            return events
        while select.select([sys.stdin], [], [], 0)[0]:
            key = sys.stdin.read(1)
            if key == "\x03":
                raise KeyboardInterrupt
            event = self.KEYS.get(key.lower())
            if event is not None:
                events.append(event)
        return events

    def wait_for(self, wanted: str) -> None:
        """Block until `wanted` is pressed (Ctrl+C still interrupts)."""
        while True:
            if wanted in self.poll():
                return
            time.sleep(0.05)


# --- Rewind history ---


class RewindHistory:
    """Per-chunk archive of the follower joint commands actually sent.

    Franka caches EE poses in its control node; here the commands themselves are
    the history, so reverse replay is exact and needs no kinematics.
    """

    def __init__(self, history_size: int) -> None:
        self._chunks: deque[list[np.ndarray]] = deque(maxlen=max(1, history_size))
        self._building: list[np.ndarray] = []
        self._anchor: Optional[np.ndarray] = None

    def reset(self, anchor: np.ndarray) -> None:
        self._chunks.clear()
        self._building = []
        self._anchor = np.asarray(anchor, dtype=np.float64).copy()

    def record(self, action: np.ndarray) -> None:
        self._building.append(np.asarray(action, dtype=np.float64).copy())

    def commit_chunk(self) -> None:
        if self._building:
            self._chunks.append(self._building)
            self._building = []

    def drop_building(self) -> None:
        """Forget the chunk in flight (used when a chunk is abandoned)."""
        self._building = []

    @property
    def available(self) -> int:
        return len(self._chunks)

    def tip(self) -> Optional[np.ndarray]:
        if self._chunks:
            return self._chunks[-1][-1]
        return self._anchor

    def pop_chunk(self) -> Optional[list[np.ndarray]]:
        if not self._chunks:
            return None
        return self._chunks.pop()


# --- Leader takeover ---


class LeaderTakeover:
    """Relative-anchored whole-body takeover with the Dobot leader arms.

    Same mapping the franka Gello controller uses, applied in joint space:
    ``follower_origin + leader_now - leader_origin``.  Unlike the old per-arm
    recording-button intervention, both arms are handed over together so one
    chunk carries a single coherent human action.
    """

    def __init__(self, leader: BimanualAgent) -> None:
        self.leader = leader
        self._active = False
        self._leader_origin = np.zeros(ACTION_DIM, dtype=np.float64)
        self._follower_origin = np.zeros(ACTION_DIM, dtype=np.float64)

    @property
    def active(self) -> bool:
        return self._active

    def _read_leader(self) -> Optional[np.ndarray]:
        try:
            value = np.asarray(self.leader.act({}), dtype=np.float64).reshape(-1)
        except Exception as exc:
            print(f"[接管] 读取主手失败：{exc}")
            return None
        if value.shape != (ACTION_DIM,) or not np.isfinite(value).all():
            print(f"[接管] 主手读数非法：shape={value.shape}")
            return None
        return value

    def activate(self, follower_now: np.ndarray) -> bool:
        leader_now = self._read_leader()
        if leader_now is None:
            return False
        self.leader.set_torque(2, False)
        # Re-read after unlocking: releasing torque can shift the arms slightly.
        settled = self._read_leader()
        self._leader_origin = (settled if settled is not None else leader_now).copy()
        self._follower_origin = np.asarray(follower_now, dtype=np.float64).copy()
        self._active = True
        return True

    def reanchor(self, follower_now: np.ndarray) -> bool:
        if not self._active:
            return False
        leader_now = self._read_leader()
        if leader_now is None:
            return False
        self._leader_origin = leader_now.copy()
        self._follower_origin = np.asarray(follower_now, dtype=np.float64).copy()
        return True

    def deactivate(self) -> None:
        if self._active:
            self.leader.set_torque(2, True)
        self._active = False

    def current_action(self) -> Optional[np.ndarray]:
        if not self._active:
            return None
        leader_now = self._read_leader()
        if leader_now is None:
            return None
        return self._follower_origin + leader_now - self._leader_origin


@dataclass
class TakeoverChunk:
    observation: dict[str, Any]
    actions: np.ndarray
    next_observation: dict[str, Any]


class TakeoverChunkBuffer:
    """Slice a continuous human demonstration into server-sized chunks."""

    def __init__(self, chunk_length: int) -> None:
        self.chunk_length = max(1, int(chunk_length))
        self._observation: Optional[dict[str, Any]] = None
        self._actions: list[np.ndarray] = []

    @property
    def needs_start_observation(self) -> bool:
        return self._observation is None

    @property
    def count(self) -> int:
        return len(self._actions)

    @property
    def full(self) -> bool:
        return len(self._actions) >= self.chunk_length

    def start(self, observation: dict[str, Any]) -> None:
        self._observation = observation
        self._actions = []

    def append_action(self, action: np.ndarray) -> None:
        self._actions.append(np.asarray(action, dtype=np.float64).copy())

    def close(self, next_observation: dict[str, Any]) -> Optional[TakeoverChunk]:
        if self._observation is None or not self._actions:
            return None
        actions = np.stack(self._actions, axis=0)
        chunk = TakeoverChunk(self._observation, actions, next_observation)
        self._observation = None
        self._actions = []
        return chunk

    def clear(self) -> None:
        self._observation = None
        self._actions = []


@dataclass
class Args:
    # --- server ---
    server_host: str = "127.0.0.1"
    server_port: int = 18000
    ssh_host: Optional[str] = None
    ssh_jump: Optional[str] = None
    remote_port: int = 8000
    ssh_timeout: float = 30.0
    # --- robot ---
    robot_port: int = 6001
    hostname: str = "127.0.0.1"
    instruction: str = "cook vegetable"
    control_hz: float = 10.0
    crop_top_camera: bool = False
    dry_run: bool = False
    # --- rewards (franka defaults) ---
    success_reward: float = 1.0
    failure_reward: float = 0.0
    fault_terminal_reward: float = -0.3
    progress_reward: float = 0.5
    small_progress_reward: float = 0.1
    regress_reward: float = -0.5
    exploration_noise_sigma: float = -1.0
    max_episode_chunks: int = 0
    # --- rewind ---
    rewind_enabled: bool = True
    rewind_history_size: int = 12
    rewind_exit_reward: float = 0.0
    rewind_prefix_reward: float = 0.1
    # Reverse replay rate as a fraction of control_hz; franka uses a 0.08
    # dynamics factor on its EE controller, this is the joint-space analogue.
    rewind_speed_scale: float = 0.5
    # --- takeover ---
    takeover_enabled: bool = True
    # --- misc ---
    pause_after_episode: bool = True
    fallback_chunk_length: int = 16


class DobotRLTClient:
    def __init__(self, args: Args, policy: Any) -> None:
        self.args = args
        self.policy = policy

        metadata = policy.get_server_metadata() or {}
        protocol = metadata.get("protocol")
        if protocol != RLT_PROTOCOL:
            raise RuntimeError(
                f"服务器协议是 {protocol!r}，本客户端只支持 {RLT_PROTOCOL!r}"
            )
        self.chunk_length = int(metadata.get("chunk_length", args.fallback_chunk_length))
        self.action_dim = int(metadata.get("action_dim", ACTION_DIM))
        if self.action_dim != ACTION_DIM:
            raise RuntimeError(
                f"服务器 action_dim={self.action_dim}，Dobot 需要 {ACTION_DIM}"
            )
        self.replay_action_space = str(metadata.get("replay_action_space", "robot"))
        self.metadata = dict(metadata)

        # The Dobot client commands joint targets.  An ee_pose server would
        # accept the same 14 numbers and mean something completely different.
        server_action_space = str(metadata.get("action_space", "joint"))
        if server_action_space != "joint":
            raise RuntimeError(
                f"服务器 action_space={server_action_space!r}，本客户端只发关节目标；"
                "请用 joint 空间的配置启动服务器"
            )
        proprio_dim = int(metadata.get("proprio_dim", ACTION_DIM))
        if proprio_dim != ACTION_DIM:
            raise RuntimeError(
                f"服务器 proprio_dim={proprio_dim}，Dobot 观测是 {ACTION_DIM} 维关节"
            )

        print("[RLT] 服务器握手信息：")
        for key in (
            "protocol", "run_name", "action_space", "action_dim", "proprio_dim",
            "chunk_length", "replay_action_space", "action_selection_mode",
            "edit_scale", "expo_num_base_samples", "use_preference_loss",
            "warmup_steps", "eval_every", "eval_only", "vla_only",
        ):
            if key in metadata:
                print(f"    {key:22} = {metadata[key]!r}")
        unknown = sorted(set(metadata) - {
            "protocol", "run_name", "action_space", "action_dim", "proprio_dim",
            "chunk_length", "replay_action_space", "action_selection_mode",
            "edit_scale", "expo_num_base_samples", "use_preference_loss",
            "warmup_steps", "eval_every", "eval_only", "vla_only",
            "transition_action_spaces",
        })
        if unknown:
            print(f"    (其他字段: {', '.join(unknown)})")
        if metadata.get("eval_only") or metadata.get("vla_only"):
            print(
                "[RLT] 注意：服务器处于 eval_only/vla_only 模式，不会训练；"
                "此时不要接管，只用 s/f 标注"
            )
        print(
            "[RLT] 归一化由服务器负责；线上动作恒为真机空间"
            f"（{ACTION_DIM} 维绝对关节，弧度；夹爪 6/13 为 [0,1]）"
        )

        self.prompt = args.instruction
        self.keyboard = KeyboardControl()
        self.rewind = RewindHistory(args.rewind_history_size)

        init_cameras(crop_top_camera=args.crop_top_camera)
        robot_client = ZMQClientRobot(port=args.robot_port, host=args.hostname)
        self.env = RobotEnv(robot_client)
        if not args.dry_run:
            for channel in (1, 2, 3):
                self.env.set_do_status([channel, 0])

        self.leader = make_leader_agent()
        self.leader.set_torque(2, True)
        self.takeover = LeaderTakeover(self.leader)
        self.takeover_buffer = TakeoverChunkBuffer(self.chunk_length)

        self.last_action: Optional[np.ndarray] = None
        self.last_observation: Optional[dict[str, Any]] = None
        self.robot_faulted = False

        # Deferred requests: b/i only take effect once the current chunk ends.
        self._pending_rewind_toggle = False
        self._pending_takeover_toggle = False
        self._rewind_mode_active = False
        self._rewind_chunks_taken = 0
        self._rewind_credit_chunks_marked = 0
        self._rewind_selection_mode: Optional[str] = None

        self._queued_reward_signal: Optional[str] = None
        self._queued_small_progress = 0.0
        self._queued_fault_reason: Optional[str] = None

        self._episode_idx = 0
        self._reset_episode_stats()

    def close(self) -> None:
        try:
            self.takeover.deactivate()
        except Exception:
            pass
        self.keyboard.stop()
        try:
            self.leader.set_torque(2, True)
        except Exception:
            pass

    def _reset_episode_stats(self) -> None:
        self._episode_reward = 0.0
        self._episode_chunks = 0
        self._episode_steps = 0
        self._episode_success = False
        self._episode_interventions = 0
        self._episode_intervention_actions = 0

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

    def observation(self) -> dict[str, Any]:
        images = get_current_images()
        state = np.asarray(self.env.get_obs()["joint_positions"], dtype=np.float32).copy()
        if self.last_action is not None:
            # Commanded gripper, not measured, matching the collection pipeline.
            for index in GRIPPER_INDICES:
                state[index] = self.last_action[index]
        observation = {
            "state": state,
            "images": {
                "cam_high": np.ascontiguousarray(images[0].transpose(2, 0, 1), dtype=np.uint8),
                "cam_left_wrist": np.ascontiguousarray(images[1].transpose(2, 0, 1), dtype=np.uint8),
                "cam_right_wrist": np.ascontiguousarray(images[2].transpose(2, 0, 1), dtype=np.uint8),
            },
            "prompt": self.prompt,
        }
        self.last_observation = observation
        return observation

    def safe_observation(self) -> dict[str, Any]:
        try:
            return self.observation()
        except Exception:
            if self.last_observation is None:
                raise
            return self.last_observation

    def prepare_action(self, proposed: np.ndarray) -> np.ndarray:
        action = np.asarray(proposed, dtype=np.float64).reshape(-1).copy()
        if action.shape != (ACTION_DIM,) or not np.isfinite(action).all():
            raise ValueError(
                f"Invalid action: shape={action.shape}, finite={np.isfinite(action).all()}"
            )
        for index in GRIPPER_INDICES:
            action[index] = np.clip(action[index], 0.0, 1.0)
        return action

    def apply_action(self, proposed: np.ndarray, *, record: bool = True) -> np.ndarray:
        action = self.prepare_action(proposed)
        if not self.args.dry_run:
            self.env.step(action, np.array([1, 1]))
        self.last_action = action.copy()
        if record:
            self.rewind.record(action)
        return action

    # --- keyboard dispatch ---

    def _dispatch(self, events: list[str]) -> Optional[str]:
        """Fold keypresses into queued state; return a terminal signal if any."""
        terminal: Optional[str] = None
        for event in events:
            if event in ("success", "failure"):
                terminal = event
                self._queued_reward_signal = event
            elif event == "progress":
                self._queued_reward_signal = "progress"
                print("[奖励] progress +%.2f" % self.args.progress_reward)
            elif event == "progress_small":
                self._queued_small_progress += float(self.args.small_progress_reward)
                self._queued_reward_signal = "progress_small"
                print(f"[奖励] small progress 累计 +{self._queued_small_progress:.2f}")
            elif event == "regress":
                self._queued_reward_signal = "regress"
                print("[奖励] regress %.2f" % self.args.regress_reward)
            elif event == "rewind_mode":
                if self.args.rewind_enabled:
                    self._pending_rewind_toggle = not self._pending_rewind_toggle
                    print(f"[倒车] 模式切换请求（本 chunk 结束后生效）")
            elif event == "takeover":
                if self.args.takeover_enabled:
                    self._pending_takeover_toggle = not self._pending_takeover_toggle
                    print("[接管] 切换请求（本 chunk 结束后生效）")
            elif event in ("rewind_step", "rewind_credit"):
                print(f"[倒车] {event} 只在倒车模式下有效，先按 b")
            elif event == "next_episode":
                pass
        return terminal

    # --- rewind ---

    def _execute_one_rewind(self) -> bool:
        """Reverse-play one finished chunk: a[n-2] -> ... -> a[0] -> previous tip.

        Stepping back through the recorded commands rather than jumping to the
        previous chunk end keeps every hop one control step wide, which is what
        makes this safe without any extra interpolation.
        """
        chunk = self.rewind.pop_chunk()
        if chunk is None:
            print("[倒车] 没有可回退的 chunk")
            return False
        final_tip = self.rewind.tip()
        if final_tip is None:
            print("[倒车] 没有锚点，无法回退")
            return False

        waypoints = list(reversed(chunk[:-1])) + [np.asarray(final_tip, dtype=np.float64)]
        span = float(np.max(np.abs(np.asarray(chunk[-1]) - np.asarray(final_tip))))
        interval = 1.0 / max(1e-3, self.args.control_hz * max(0.05, self.args.rewind_speed_scale))
        print(
            f"[倒车] 反向回放 chunk steps={len(chunk)} -> {len(waypoints)} 次运动 "
            f"(最大关节跨度 {np.rad2deg(span):.1f}deg) | 剩余 chunks={self.rewind.available}"
        )
        deadline = time.monotonic()
        for waypoint in waypoints:
            # record=False: rewinding must not append to the history it consumes.
            self.apply_action(waypoint, record=False)
            deadline += interval
            time.sleep(max(0.0, deadline - time.monotonic()))
        self._rewind_chunks_taken += 1
        return True

    def _apply_rewind_exit_correction(self) -> None:
        reward = float(self.args.rewind_exit_reward)
        chunks = int(self._rewind_chunks_taken)
        credit_chunks = int(self._rewind_credit_chunks_marked)
        selection_mode = self._rewind_selection_mode
        self._rewind_chunks_taken = 0
        self._rewind_credit_chunks_marked = 0
        self._rewind_selection_mode = None
        if reward == 0.0:
            return
        try:
            if selection_mode == "credit" and credit_chunks > 0:
                request = {
                    "rlt/request": "rewind_credit_correction",
                    "terminal_reward": reward,
                    "bad_chunks": credit_chunks,
                    "prefix_reward": float(self.args.rewind_prefix_reward),
                }
            elif selection_mode == "physical" and chunks > 0:
                request = {
                    "rlt/request": "rewind_exit_correction",
                    "terminal_reward": reward,
                    "chunks_rewound": chunks,
                }
            else:
                return
            response = self.policy.infer(request)
            if response.get("applied"):
                self._episode_reward += float(response.get("episode_reward_delta", 0.0))
            print(
                f"[倒车] 修正 applied={response.get('applied')} "
                f"mode={selection_mode} chunks={chunks or credit_chunks} "
                f"bad_start={response.get('bad_branch_start_index')} "
                f"bad_terminal={response.get('bad_terminal_index')} "
                f"anchor={response.get('replacement_anchor_index')}"
            )
        except Exception as exc:
            print(f"[倒车] 修正失败：{exc}")

    def _service_rewind_mode(self) -> None:
        """Block the policy while the operator rewinds with r / q, exit on b."""
        self._rewind_mode_active = True
        print(
            f"[倒车] 模式 ON：r=物理回退一个 chunk，q=只标记坏 chunk（不动），"
            f"b=退出 | 可回退 {self.rewind.available} 个 chunk"
        )
        while True:
            events = self.keyboard.poll()
            for event in events:
                if event == "rewind_step":
                    if self._rewind_selection_mode == "credit":
                        print("[倒车] 已在 q 标记模式，不能混用 r")
                        continue
                    self._rewind_selection_mode = "physical"
                    try:
                        self._execute_one_rewind()
                    except Exception as exc:
                        self.robot_faulted = True
                        print(f"[倒车] 回退失败：{exc}")
                        self._pending_rewind_toggle = False
                        self._rewind_mode_active = False
                        return
                elif event == "rewind_credit":
                    if self._rewind_selection_mode == "physical":
                        print("[倒车] 已在 r 物理回退模式，不能混用 q")
                        continue
                    if self._rewind_credit_chunks_marked >= self._episode_chunks:
                        print("[倒车] 没有更早的 policy chunk 可标记")
                        continue
                    self._rewind_selection_mode = "credit"
                    self._rewind_credit_chunks_marked += 1
                    print(
                        f"[倒车] 信用标记 {self._rewind_credit_chunks_marked} 个 chunk（机械臂不动）"
                    )
                elif event == "rewind_mode":
                    self._pending_rewind_toggle = False
                    self._rewind_mode_active = False
                    self._apply_rewind_exit_correction()
                    print("[倒车] 模式 OFF，恢复 policy")
                    return
                elif event == "takeover":
                    print("[倒车] 倒车模式下不能切换接管，先按 b 退出")
            time.sleep(0.03)

    # --- server calls ---

    def _act(self, observation: dict[str, Any]) -> dict[str, Any]:
        request: dict[str, Any] = {
            "rlt/request": "act",
            "observation": observation,
        }
        if self.args.exploration_noise_sigma >= 0.0:
            request["exploration_noise_sigma"] = float(self.args.exploration_noise_sigma)
        return self.policy.infer(request)

    def _rewards_array(self, steps_executed: int, signal: Optional[str]) -> tuple[np.ndarray, bool, float, dict[str, Any]]:
        rewards = np.zeros(self.chunk_length, dtype=np.float32)
        index = max(0, min(self.chunk_length, max(steps_executed, 1)) - 1)
        done = False
        bootstrap_mask = 1.0
        info: dict[str, Any] = {}
        if signal == "progress":
            rewards[index] = float(self.args.progress_reward)
            info["progress"] = True
        elif signal == "progress_small":
            rewards[index] = float(self._queued_small_progress)
            info["progress"] = True
            info["progress_small"] = True
        elif signal == "regress":
            rewards[index] = float(self.args.regress_reward)
            info["regress"] = True
        elif signal == "success":
            rewards[index] = float(self.args.success_reward)
            done = True
            info["success"] = True
        elif signal == "failure":
            rewards[index] = float(self.args.failure_reward)
            done = True
            info["success"] = False
        elif signal == "fault":
            rewards[index] = float(self.args.fault_terminal_reward)
            done = True
            bootstrap_mask = 0.0
            info["success"] = False
            info["fault"] = True
            info["fault_reason"] = str(self._queued_fault_reason or "dobot_fault")
        self._queued_reward_signal = None
        self._queued_small_progress = 0.0
        self._queued_fault_reason = None
        return rewards, done, bootstrap_mask, info

    def _send_transition(
        self,
        transition_id: str,
        next_observation: dict[str, Any],
        action_chunk: np.ndarray,
        steps_executed: int,
        signal: Optional[str],
        *,
        intervention: bool,
    ) -> bool:
        rewards, done, bootstrap_mask, extra = self._rewards_array(steps_executed, signal)
        info: dict[str, Any] = {
            "steps_executed": steps_executed,
            "client": "run_rlt_dobot_client",
            "intervention": intervention,
        }
        info.update(extra)
        if intervention:
            info["leader_takeover"] = True
        if (
            self.args.max_episode_chunks > 0
            and self._episode_chunks + 1 >= self.args.max_episode_chunks
            and not done
        ):
            done = True
            info["success"] = False
            info["timeout"] = True

        response = self.policy.infer(
            {
                "rlt/request": "transition",
                "transition_id": str(transition_id),
                "next_observation": next_observation,
                "rewards": rewards,
                "done": bool(done),
                "bootstrap_mask": float(bootstrap_mask),
                "info": info,
                "intervention": bool(intervention),
                "action_chunk": action_chunk,
                # The client only ever knows real-robot joint targets.  The
                # server normalizes for its replay buffer when it needs to.
                "action_chunk_space": "robot",
            }
        )
        self._episode_chunks += 1
        self._episode_steps += steps_executed
        self._episode_reward += float(rewards.sum())
        if intervention:
            self._episode_interventions += 1
            self._episode_intervention_actions += steps_executed
        if done:
            self._episode_success = bool(info.get("success", False))
        print(
            f"[transition] steps={steps_executed} reward={float(rewards.sum()):.2f} "
            f"done={done} bootstrap={bootstrap_mask:.1f} intervention={intervention} "
            f"buffer={response.get('buffer_size')} updates={response.get('total_updates')}"
        )
        return done

    def _send_episode_end(self) -> dict[str, Any]:
        stats = {
            "episode_reward": float(self._episode_reward),
            "episode_chunks": int(self._episode_chunks),
            "episode_steps": int(self._episode_steps),
            "episode_interventions": int(self._episode_interventions),
            "episode_intervention_actions": int(self._episode_intervention_actions),
            "success": bool(self._episode_success),
        }
        response: dict[str, Any] = {}
        try:
            response = self.policy.infer({"rlt/request": "episode_end", "stats": stats})
            print(
                f"[episode {self._episode_idx}] success={stats['success']} "
                f"reward={stats['episode_reward']:.2f} chunks={stats['episode_chunks']} "
                f"steps={stats['episode_steps']} "
                f"interventions={stats['episode_interventions']} "
                f"buffer={response.get('buffer_size')} updates={response.get('total_updates')} "
                f"checkpoint={response.get('checkpoint')}"
            )
        except Exception as exc:
            print(f"[episode_end] 失败：{exc}")
        finally:
            self._episode_idx += 1
            self._reset_episode_stats()
        return response

    # --- chunk execution ---

    def _execute_policy_chunk(self, actions: np.ndarray) -> tuple[np.ndarray, int, Optional[str]]:
        interval = 1.0 / self.args.control_hz
        executed: list[np.ndarray] = []
        signal: Optional[str] = None
        deadline = time.monotonic()
        for action in actions:
            executed.append(self.apply_action(action))
            terminal = self._dispatch(self.keyboard.poll())
            if terminal is not None:
                signal = terminal
                break
            deadline += interval
            time.sleep(max(0.0, deadline - time.monotonic()))
            if time.monotonic() - deadline > interval:
                deadline = time.monotonic()
        if signal is None:
            signal = self._queued_reward_signal
        # Pad to the wire-fixed chunk length by repeating the last command.
        chunk = np.repeat(executed[-1][None], len(actions), axis=0)
        for index, action in enumerate(executed):
            chunk[index] = action
        self.rewind.commit_chunk()
        return chunk, len(executed), signal

    def _run_takeover(self, observation: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        """Drive the follower from the leader, uploading one chunk at a time.

        Returns (episode_done, latest_observation).
        """
        assert self.last_action is not None
        if not self.takeover.activate(self.last_action):
            print("[接管] 启动失败，继续用 policy")
            self._pending_takeover_toggle = False
            return False, observation

        print(
            f"[接管] 已接管双臂；再按 i 归还 policy。每 {self.chunk_length} 步上传一个 chunk"
        )
        interval = 1.0 / self.args.control_hz
        self.takeover_buffer.clear()
        self.takeover_buffer.start(observation)
        deadline = time.monotonic()
        done = False
        latest = observation
        try:
            while True:
                target = self.takeover.current_action()
                if target is None:
                    print("[接管] 主手读数丢失，归还 policy")
                    break
                try:
                    self.apply_action(target)
                except Exception as exc:
                    self.robot_faulted = True
                    self._queued_fault_reason = f"{type(exc).__name__}: {exc}"
                    print(f"[接管] 执行失败：{exc}")
                    break
                self.takeover_buffer.append_action(self.last_action)

                terminal = self._dispatch(self.keyboard.poll())
                stop_requested = self._pending_takeover_toggle is False

                if terminal is not None or self.takeover_buffer.full or stop_requested:
                    latest = self.observation()
                    chunk = self.takeover_buffer.close(latest)
                    if chunk is not None:
                        self.rewind.commit_chunk()
                        done = self._upload_takeover_chunk(chunk, terminal)
                    if done or terminal is not None or stop_requested:
                        break
                    self.takeover_buffer.start(latest)

                deadline += interval
                time.sleep(max(0.0, deadline - time.monotonic()))
                if time.monotonic() - deadline > interval:
                    deadline = time.monotonic()
        finally:
            self.takeover.deactivate()
            self.takeover_buffer.clear()
            self._pending_takeover_toggle = False
            print("[接管] 已归还 policy")
        return done, latest

    def _upload_takeover_chunk(
        self, chunk: TakeoverChunk, signal: Optional[str]
    ) -> bool:
        result = self._act(chunk.observation)
        if result.get("should_stop", False):
            print("[接管] 服务器要求停止")
            return True
        padded = np.repeat(chunk.actions[-1][None], self.chunk_length, axis=0)
        for index in range(min(len(chunk.actions), self.chunk_length)):
            padded[index] = chunk.actions[index]
        return self._send_transition(
            str(result["transition_id"]),
            chunk.next_observation,
            padded,
            int(min(len(chunk.actions), self.chunk_length)),
            signal,
            intervention=True,
        )

    # --- episode ---

    def _prepare_episode(self) -> dict[str, Any]:
        self.keyboard.stop()
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
        input(f"请布置第 {self._episode_idx} 回合场景，确认安全后按 Enter：")

        obs = self.env.get_obs()
        self.last_action = np.asarray(obs["joint_positions"], dtype=np.float64).copy()
        for index in GRIPPER_INDICES:
            self.last_action[index] = 1.0
        self.rewind.reset(self.last_action)
        self._pending_rewind_toggle = False
        self._pending_takeover_toggle = False
        self._rewind_mode_active = False
        self._rewind_chunks_taken = 0
        self._rewind_credit_chunks_marked = 0
        self._rewind_selection_mode = None
        self._queued_reward_signal = None
        self._queued_small_progress = 0.0
        self._queued_fault_reason = None

        observation = self.observation()
        self.keyboard.start()
        print(
            f"\n=== 回合 {self._episode_idx} 开始 ===\n"
            "s/空格=成功  f=失败  p=进展(+0.5)  o=小进展(+0.1,可叠加)  x=倒退(-0.5)\n"
            "b=倒车模式  r=回退一个chunk  q=只标记坏chunk  i=主手接管  Ctrl+C=停止"
        )
        return observation

    def run_episode(self) -> bool:
        """Run one episode. Returns True if the server asked us to stop."""
        observation = self._prepare_episode()
        should_stop = False
        while True:
            if self._pending_rewind_toggle and not self._rewind_mode_active:
                self._service_rewind_mode()
                observation = self.safe_observation()
                continue

            if self._pending_takeover_toggle:
                done, observation = self._run_takeover(observation)
                if done:
                    break
                continue

            try:
                result = self._act(observation)
            except Exception as exc:
                print(f"[act] 失败：{exc}")
                raise
            if result.get("should_stop", False):
                print("[RLT] 服务器要求停止")
                should_stop = True
                break

            actions = np.asarray(result["actions"], dtype=np.float64)
            if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
                raise RuntimeError(f"服务器返回动作形状非法：{actions.shape}")

            try:
                executed_chunk, steps, signal = self._execute_policy_chunk(actions)
            except Exception as exc:
                self.robot_faulted = True
                self._queued_fault_reason = f"{type(exc).__name__}: {exc}"
                print(f"[执行] 机械臂故障：{exc}")
                self.rewind.drop_building()
                next_observation = self.safe_observation()
                held = np.repeat(
                    np.asarray(self.last_action, dtype=np.float64)[None],
                    self.chunk_length, axis=0,
                )
                self._send_transition(
                    str(result["transition_id"]), next_observation, held, 1, "fault",
                    intervention=False,
                )
                break

            next_observation = self.observation()
            done = self._send_transition(
                str(result["transition_id"]), next_observation,
                executed_chunk, steps, signal,
                intervention=False,
            )
            observation = next_observation
            if done:
                break

        self.keyboard.stop()
        response = self._send_episode_end()
        if response.get("should_stop", False):
            should_stop = True
        if not should_stop and self.args.pause_after_episode:
            self.keyboard.start()
            print("回合结束。按 n 开始下一回合（Ctrl+C 停止）")
            try:
                self.keyboard.wait_for("next_episode")
            finally:
                self.keyboard.stop()
        return should_stop


def main(args: Args) -> int:
    if args.control_hz <= 0:
        raise ValueError("control_hz must be positive")

    tunnel: Optional[SSHTunnel] = None
    if args.ssh_host:
        if args.server_host not in ("127.0.0.1", "localhost"):
            raise ValueError("--ssh-host 时 --server-host 必须是 127.0.0.1")
        tunnel = SSHTunnel(
            host=normalize_ssh_target(args.ssh_host),
            local_port=args.server_port,
            remote_port=args.remote_port,
            jump=normalize_ssh_target(args.ssh_jump) if args.ssh_jump else None,
            connect_timeout=args.ssh_timeout,
        )
        # Fail before the cameras and the robot are brought up.
        tunnel.ensure()

    client: Optional[DobotRLTClient] = None
    try:
        print(f"连接 RLT 服务器 {args.server_host}:{args.server_port} ...")
        policy = _websocket_client_policy.WebsocketClientPolicy(
            host=args.server_host, port=args.server_port
        )
        client = DobotRLTClient(args, policy)
        while True:
            if client.run_episode():
                print("[RLT] 服务器已达停止条件")
                break
    except KeyboardInterrupt:
        print("\n停止 RLT Dobot 客户端")
        return 0
    finally:
        if client is not None:
            client.close()
        stop_cameras()
        cv2.destroyAllWindows()
        if tunnel is not None:
            tunnel.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(tyro.cli(Args)))
