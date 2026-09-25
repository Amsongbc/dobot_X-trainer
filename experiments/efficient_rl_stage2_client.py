#!/usr/bin/env python3
"""Standalone RLT Stage 2 Dobot client with cached-chunk rewind and takeover.

This is the Dobot counterpart of ``franka_inference_node_rlt.py``: it speaks the
same client-driven RLT websocket protocol that ``exp/stage2_server_shuo.sh``
serves, so the same Stage 2 server can drive either arm.

    act(obs) -> execute chunk -> transition(next_obs, reward, done)

The Franka stack splits that work across two processes: the inference node owns
the protocol and the ROS control node owns the rewind chunk history and the
reverse playback.  The Dobot has no such control node -- this file executes the
chunk itself -- so the rewind history, the reverse playback and the takeover
loop all live here, driven directly by the keyboard instead of by
``/franka/rewind_mode`` and ``/franka/rewind_step``.

Keyboard labels while this process is focused:
    s or Space : success, reward=+success_reward, end episode
    f          : failure, reward=failure_reward, end episode
    p          : progress, reward=+progress_reward, continue
    o          : small progress, +small_progress_reward per press (stacks)
    x          : regress, reward=regress_reward, continue
    n          : after an episode end, start the next episode
    b          : enter/cancel rewind mode after the current chunk finishes
    r          : while in rewind mode, reverse-play one finished action chunk
    q          : mark one previous replay chunk as bad without moving the robot
    i          : optional keyboard takeover shortcut
    a          : switch VLA <-> actor from the next chunk; only actor chunks
                 (and takeovers while the actor is on) are written to replay
    Ctrl+C     : stop

After ``b`` pauses policy in rewind mode, either leader recording button starts
takeover directly from the paused pose. Teleoperation keeps running at
``control_hz``, while held-button actions are sampled into replay at
``takeover_record_hz``. Releasing the button holds the current pose and keeps
takeover active. Press ``b`` again to return to policy. No ``i`` press is
required for this workflow.

Leaving rewind via b/i can penalize the discarded branch and cut its TD
bootstrap; see --rewind-physical-exit-reward / --rewind-credit-exit-reward.

This file is self-contained: the keyboard monitor, the intervention chunk
buffer, the rewind history, the RealSense capture threads and the Dobot
environment client are all inlined, so it drives the robot without importing
any other experiments module.

SSH forwarding (same options as run_stage2_env_client.py; no second terminal):
    python efficient_rl_stage2_client.py \
        --ssh-host sysu_xdliang_2@pytorch-ng-30000 \
        --ssh-jump 13c09e665f284d65a28a7545f79197c7@proxy.nscc-gz.cn:8022
Local 127.0.0.1:18000 forwards to the server's 127.0.0.1:8000. Both ports
can be overridden with --server-port / --remote-port. Without --ssh-host,
connect directly to --server-host:--server-port as before.
"""
from __future__ import annotations

import dataclasses
import os
import select
import shutil
import socket
import subprocess
import sys
import tempfile
import termios
import threading
import time
import traceback
import tty
from collections import deque
from dataclasses import dataclass
from typing import Any, Optional

import cv2
import numpy as np
import tyro
from websockets.exceptions import ConnectionClosed, InvalidHandshake
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
NETWORK_ERRORS = (ConnectionError, OSError, ConnectionClosed, InvalidHandshake)


# --- Dobot hardware helpers (inlined; previously experiments/hw_utils.py) ---
# Frames are published as RGB, matching what the model was trained on.  This
# differs from the older clients, which passed the collection script's
# cv2.imwrite-oriented BGR flip straight through to the server.

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


# --- SSH tunnel (so the client needs no second terminal) ---


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
        """Probe by binding, not by connecting.

        Connecting would open a real websocket connection through the tunnel
        that the Stage 2 server would see and immediately lose. SO_REUSEADDR
        ignores TIME_WAIT sockets, while an active listener still blocks bind.
        """
        probe = socket.socket()
        try:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
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
        """Start the tunnel, or rebuild it if the ssh child has died."""
        if self._adopted:
            return
        if self._process is not None and self._process.poll() is None:
            return
        if self._process is not None:
            print(f"SSH 隧道已断开（ssh 退出码 {self._process.returncode}），正在重建 ...")
        elif self._port_taken():
            # A just-terminated ssh process can keep the port unavailable for
            # a short time.  Do not permanently mistake that transient state
            # for a reusable tunnel.
            deadline = time.monotonic() + 1.0
            while self._port_taken() and time.monotonic() < deadline:
                time.sleep(0.1)
            if self._port_taken():
                print(
                    f"检测到 127.0.0.1:{self.local_port} 持续被占用，"
                    "复用现有转发"
                )
                self._adopted = True
                return
            print(
                f"127.0.0.1:{self.local_port} 的短暂占用已释放，"
                "将建立新 SSH 隧道"
            )
        self._start()

    def invalidate_adopted(self) -> None:
        """Retry ownership when a supposedly reusable forward cannot connect."""
        if self._adopted:
            print("现有 SSH 转发不可用，下次重连将重新检测或建立隧道")
            self._adopted = False

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
        raise RuntimeError(
            f"SSH 隧道 {self.connect_timeout:.0f}s 内未就绪：{message}"
        )

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


# --- Keyboard (edge-triggered, mirrors franka_inference_node_rlt) ---


class KeyboardMonitor:
    """Edge-triggered keyboard reader.

    Reward keys (s/f/p/o/x/n) are latched and consumed by the transition that
    follows; mode keys (b/r/q/i) are delivered immediately to the client so a
    rewind step can run while no chunk is in flight.
    """

    REWARD_KEYS = {
        "s": "success",
        " ": "success",
        "f": "failure",
        "p": "progress",
        "o": "progress_small",
        "x": "regress",
        "n": "next",
    }

    def __init__(
        self,
        *,
        rewind_mode_key: str = "b",
        rewind_step_key: str = "r",
        rewind_credit_key: str = "q",
        takeover_key: str = "i",
        actor_switch_key: str = "",
    ) -> None:
        self._old_settings = None
        self._raw = False
        self._lock = threading.Lock()
        self._reward_signal: Optional[str] = None
        self._small_progress_presses = 0
        self._edge_queue: deque[str] = deque()
        # An empty key disables that edge signal.
        self._edges = {}
        for key, name in (
            (rewind_mode_key, "rewind_mode"),
            (rewind_step_key, "rewind_step"),
            (rewind_credit_key, "rewind_credit"),
            (takeover_key, "takeover"),
            (actor_switch_key, "actor_switch"),
        ):
            key = (key or "").strip().lower()[:1]
            if key:
                self._edges[key] = name

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

    def poll(self) -> None:
        """Drain pending keystrokes into the latched reward / edge queues."""
        if not self._raw:
            return
        while select.select([sys.stdin], [], [], 0)[0]:
            key = sys.stdin.read(1)
            if key == "\x03":
                raise KeyboardInterrupt
            lowered = key.lower()
            if lowered in self._edges:
                with self._lock:
                    self._edge_queue.append(self._edges[lowered])
                continue
            signal = self.REWARD_KEYS.get(lowered)
            if signal is not None:
                with self._lock:
                    # o stacks: several presses before the next transition add up.
                    if signal == "progress_small":
                        if self._reward_signal != "progress_small":
                            self._small_progress_presses = 0
                        self._small_progress_presses += 1
                    else:
                        self._small_progress_presses = 0
                    self._reward_signal = signal

    def take_edge(self) -> Optional[str]:
        with self._lock:
            return self._edge_queue.popleft() if self._edge_queue else None

    def peek_reward(self) -> Optional[str]:
        with self._lock:
            return self._reward_signal

    def take_reward(self) -> tuple[Optional[str], int]:
        """Consume the latched reward and how many times ``o`` was pressed."""
        with self._lock:
            signal, self._reward_signal = self._reward_signal, None
            presses, self._small_progress_presses = self._small_progress_presses, 0
            return signal, presses

    def clear(self) -> None:
        with self._lock:
            self._reward_signal = None
            self._small_progress_presses = 0
            self._edge_queue.clear()


# --- Intervention chunk buffer (vendored from gello_chunk_buffer.py) ---


@dataclasses.dataclass(frozen=True)
class InterventionChunk:
    """One executed takeover chunk with observations from its exact boundaries."""

    observation: dict[str, Any]
    actions: np.ndarray
    next_observation: dict[str, Any]
    steps_executed: int
    step_observations: tuple[dict[str, Any], ...] = ()


class InterventionChunkBuffer:
    """Accumulate actions and freeze completed chunks at control boundaries.

    The caller records the start observation before the first action.  Once
    chunk_length actions have been published, it records the boundary
    observation immediately before publishing the next action.  That same
    boundary observation is both the previous chunk's next_observation and the
    next chunk's start observation.

    Completed chunks live in a FIFO so a slow policy server cannot overwrite
    their observations while continuous teleoperation keeps running.
    """

    def __init__(self, chunk_length: int) -> None:
        if chunk_length <= 0:
            raise ValueError("chunk_length must be positive")
        self.chunk_length = int(chunk_length)
        self._observation: Optional[dict[str, Any]] = None
        self._actions: list[np.ndarray] = []
        self._step_observations: list[dict[str, Any]] = []
        self._completed: deque[InterventionChunk] = deque()

    @property
    def needs_start_observation(self) -> bool:
        return self._observation is None

    @property
    def needs_boundary_observation(self) -> bool:
        return len(self._actions) == self.chunk_length

    @property
    def completed_count(self) -> int:
        return len(self._completed)

    @property
    def action_count(self) -> int:
        return len(self._actions)

    def clear(self) -> None:
        self._observation = None
        self._actions.clear()
        self._step_observations.clear()
        self._completed.clear()

    def start(self, observation: dict[str, Any]) -> None:
        if self._observation is not None or self._actions:
            raise RuntimeError("cannot replace the start observation of an active chunk")
        self._observation = observation

    def append_action(self, action: np.ndarray, observation: Optional[dict[str, Any]] = None) -> None:
        if self._observation is None:
            raise RuntimeError("chunk requires a start observation before its first action")
        if len(self._actions) >= self.chunk_length:
            raise RuntimeError("chunk is full and must be closed at a boundary before appending")
        if observation is not None:
            self._step_observations.append({"offset": len(self._actions), "observation": observation})
        self._actions.append(np.asarray(action, dtype=np.float32).copy())

    def _finish(
        self,
        next_observation: dict[str, Any],
        *,
        continue_from_boundary: bool,
    ) -> InterventionChunk:
        if self._observation is None:
            raise RuntimeError("cannot close a chunk without a start observation")
        steps_executed = len(self._actions)
        if not 0 < steps_executed <= self.chunk_length:
            raise RuntimeError(f"invalid intervention step count: {steps_executed}")
        actions = [entry.copy() for entry in self._actions]
        # The RLT wire format has a fixed action horizon. A recording-button
        # release may end before that horizon, so pad only the wire payload and
        # preserve the real Hz-derived length in steps_executed.
        actions.extend(
            actions[-1].copy() for _ in range(self.chunk_length - steps_executed)
        )
        chunk = InterventionChunk(
            observation=self._observation,
            actions=np.stack(actions, axis=0).astype(np.float32, copy=False),
            next_observation=next_observation,
            steps_executed=steps_executed,
            step_observations=tuple(self._step_observations),
        )
        self._completed.append(chunk)
        self._observation = next_observation if continue_from_boundary else None
        self._actions.clear()
        self._step_observations.clear()
        return chunk

    def close_at_boundary(self, next_observation: dict[str, Any]) -> InterventionChunk:
        if len(self._actions) != self.chunk_length:
            raise RuntimeError(
                f"cannot close an incomplete chunk: {len(self._actions)}/{self.chunk_length} actions"
            )
        # The state at one boundary is shared by adjacent full transitions:
        # next_observation_t == observation_{t+1}.
        return self._finish(next_observation, continue_from_boundary=True)

    def close_on_release(self, next_observation: dict[str, Any]) -> InterventionChunk:
        """Finish real recording-button steps and pad only the wire payload."""
        return self._finish(next_observation, continue_from_boundary=False)

    def pop_completed(self) -> Optional[InterventionChunk]:
        if not self._completed:
            return None
        return self._completed.popleft()


# --- Rewind history (this file's port of the Franka control node's cache) ---


class RewindHistory:
    """Cache finished action chunks so whole chunks can be reverse-played.

    The Franka stack keeps this inside ``franka_control_node.py`` because that
    process owns the arm.  On the Dobot this client is the executor, so the
    cache lives next to the code that calls ``env.step``.

    Each chunk holds every joint command actually sent to the follower, plus a
    pre-chunk anchor captured before the first command of the episode, so
    rewinding the oldest chunk still has somewhere to land.
    """

    def __init__(self, history_size: int) -> None:
        self._lock = threading.Lock()
        self._size = max(1, int(history_size))
        self._chunks: deque[list[np.ndarray]] = deque(maxlen=self._size)
        self._building: list[np.ndarray] = []
        self._anchor: Optional[np.ndarray] = None

    @property
    def history_size(self) -> int:
        with self._lock:
            return self._size

    def resize(self, history_size: int) -> None:
        requested = max(1, int(history_size))
        with self._lock:
            if requested == self._size:
                return
            retained = list(self._chunks)[-requested:]
            self._size = requested
            self._chunks = deque(retained, maxlen=requested)

    def clear(self) -> None:
        with self._lock:
            self._chunks.clear()
            self._building.clear()
            self._anchor = None

    def seed(self, action: np.ndarray) -> None:
        """Capture the command pose once, before the first action of an episode."""
        entry = np.asarray(action, dtype=np.float64).reshape(-1).copy()
        with self._lock:
            if self._anchor is not None or self._building or self._chunks:
                return
            self._anchor = entry

    def record(self, action: np.ndarray) -> None:
        """Append one executed command into the in-progress chunk buffer."""
        entry = np.asarray(action, dtype=np.float64).reshape(-1).copy()
        with self._lock:
            self._building.append(entry)

    def drop_building(self) -> None:
        """Discard a partially executed chunk (fault, or takeover took over)."""
        with self._lock:
            self._building.clear()

    def commit(self) -> int:
        """Finalize the in-progress chunk once its execution has finished."""
        with self._lock:
            if not self._building:
                return 0
            chunk = list(self._building)
            self._building.clear()
            self._chunks.append(chunk)
            return len(chunk)

    def available(self) -> int:
        """How many whole-chunk rewind steps are possible."""
        with self._lock:
            count = len(self._chunks)
            if self._building:
                count += 1
            return count

    def committed_chunks(self) -> int:
        with self._lock:
            return len(self._chunks)

    def peek_tip(self) -> Optional[np.ndarray]:
        """Latest recorded command, or the pre-chunk anchor."""
        with self._lock:
            if self._building:
                return self._building[-1].copy()
            for chunk in reversed(self._chunks):
                if chunk:
                    return chunk[-1].copy()
            return None if self._anchor is None else self._anchor.copy()

    def pop_chunk(self) -> Optional[list[np.ndarray]]:
        """Remove and return the newest finished chunk."""
        with self._lock:
            if self._building:
                chunk = [entry.copy() for entry in self._building]
                self._building.clear()
                return chunk
            while self._chunks:
                chunk = self._chunks.pop()
                if chunk:
                    return [entry.copy() for entry in chunk]
            return None


@dataclass
class Args:
    # --- server ---
    server_host: str = "127.0.0.1"
    server_port: int = 18000
    # Set --ssh-host to let this process own the ssh -N -L tunnel itself.
    # Left unset, nothing changes: connect to server_host:server_port as before.
    ssh_host: Optional[str] = None
    ssh_jump: Optional[str] = None
    # Port the Stage 2 server listens on, on the far side of the tunnel.
    remote_port: int = 8000
    ssh_timeout: float = 30.0
    reconnect_delay: float = 3.0

    # --- robot ---
    robot_port: int = 6001
    hostname: str = "127.0.0.1"
    instruction: str = "pour water"
    control_hz: float = 10.0
    crop_top_camera: bool = False
    dry_run: bool = False

    # --- chunk publishing ---
    max_actions_to_publish: int = 0
    """0 = publish the whole server chunk (stop-and-go)."""
    use_last_actions: bool = False
    action_publish_interval: float = 0.0
    exploration_noise_sigma: float = -1.0
    """<0 leaves the server's configured actor_noise_sigma untouched."""
    require_normalized_replay: bool = True
    max_episode_chunks: int = 0

    # --- rewards ---
    success_reward: float = 1.0
    failure_reward: float = 0.0
    progress_reward: float = 0.5
    small_progress_reward: float = 0.1
    regress_reward: float = -0.5
    fault_terminal_reward: float = -0.3

    # --- rewind (倒车) ---
    rewind_enabled: bool = True
    rewind_mode_key: str = "b"
    rewind_step_key: str = "r"
    rewind_credit_key: str = "q"
    rewind_history_size: int = 15
    rewind_prefix_reward: float = 0.1
    rewind_physical_exit_reward: float = -0.5
    rewind_credit_exit_reward: float = -0.1
    rewind_max_step: float = 0.0004
    """Per-servo interpolation step during reverse playback; smaller = slower."""
    rewind_max_steps_per_move: int = 400

    # --- takeover (接管) ---
    takeover_enabled: bool = True
    takeover_key: str = "i"
    relative_takeover: bool = True
    """Map leader displacement onto the follower's current pose (recommended)."""
    takeover_record_hz: float = 3.0
    """Replay sampling rate for human actions; teleoperation still uses control_hz."""

    # --- actor switch (切换 actor / VLA) ---
    actor_switch_key: str = "a"
    """Toggle VLA <-> actor, effective from the next chunk. VLA chunks, and
    takeovers made while on VLA, run on the robot but are not stored."""
    start_with_actor: bool = False
    """Mode every episode starts in; False = VLA until the key is pressed."""

    pause_after_episode: bool = True


def make_leader_agent() -> BimanualAgent:
    _, hands = load_ini_data_hands()
    return BimanualAgent(
        DobotAgent(which_hand="LEFT", dobot_config=hands["HAND_LEFT"]),
        DobotAgent(which_hand="RIGHT", dobot_config=hands["HAND_RIGHT"]),
    )


class DobotStage2RLTClient:
    """Client-driven RLT loop for the Dobot with cached-chunk rewind."""

    # Keep the legacy Dobot controller's definition of an A-button short press.
    BUTTON_A_SHORT_PRESS_SECONDS = 0.5

    def __init__(self, args: Args, policy: Any) -> None:
        self.args = args
        self.policy = policy
        self.prompt = args.instruction

        metadata = policy.get_server_metadata()
        self.chunk_length = int(metadata.get("chunk_length", 0) or 0)
        if self.chunk_length <= 0:
            raise RuntimeError(f"server metadata has no usable chunk_length: {metadata}")
        server_action_dim = int(metadata.get("action_dim", ACTION_DIM) or ACTION_DIM)
        if server_action_dim != ACTION_DIM:
            raise RuntimeError(
                f"server action_dim={server_action_dim} does not match the Dobot's {ACTION_DIM}"
            )
        if 0 < int(args.max_actions_to_publish) < self.chunk_length:
            raise ValueError("This RLT server requires complete chunks; set max_actions_to_publish=0")
        self.step_obs_offsets = set(int(v) for v in metadata.get("step_obs_offsets", []))
        self.takeover_step_obs = bool(metadata.get("step_window_include_intervention", False))
        self.replay_action_space = str(metadata.get("replay_action_space", "robot"))
        self.policy_switch_supported = bool(metadata.get("supports_policy_switch", False))
        if not self.policy_switch_supported:
            print(
                "[Actor switch] 服务端不支持 policy 字段，切换键已禁用，全程 actor 且全部写入 replay；"
                "请更新服务端 online_rl_policy.py"
            )
        print(
            f"Server metadata: chunk_length={self.chunk_length} "
            f"action_dim={server_action_dim} replay_space={self.replay_action_space} "
            f"run={metadata.get('run_name')} eval_only={metadata.get('eval_only')}"
        )

        self.keyboard = KeyboardMonitor(
            rewind_mode_key=args.rewind_mode_key if args.rewind_enabled else "",
            rewind_step_key=args.rewind_step_key if args.rewind_enabled else "",
            rewind_credit_key=args.rewind_credit_key if args.rewind_enabled else "",
            takeover_key=args.takeover_key if args.takeover_enabled else "",
            actor_switch_key=args.actor_switch_key if self.policy_switch_supported else "",
        )

        # --- hardware ---
        init_cameras(crop_top_camera=args.crop_top_camera)
        robot_client = ZMQClientRobot(port=args.robot_port, host=args.hostname)
        self.env = RobotEnv(robot_client)
        if not args.dry_run:
            for channel in (1, 2, 3):
                self.env.set_do_status([channel, 0])
        self.leader = make_leader_agent()
        self.leader.set_torque(2, True)

        self.last_action: Optional[np.ndarray] = None
        self.last_observation: Optional[dict[str, Any]] = None
        self.robot_faulted = False

        self._env_lock = threading.RLock()
        self._policy_lock = threading.RLock()
        self._leader_state_lock = threading.RLock()

        # --- leader state ---
        self.was_intervening = np.array([False, False])
        self.leader_unlocked = np.array([False, False])
        self._button_a_pressed = np.array([False, False])
        self._button_a_pressed_at = np.zeros(2, dtype=np.float64)
        self.leader_origin = np.zeros(ACTION_DIM, dtype=np.float64)
        self.follower_origin = np.zeros(ACTION_DIM, dtype=np.float64)

        # --- rewind state (mirrors franka_inference_node_rlt) ---
        self.rewind_history = RewindHistory(args.rewind_history_size)
        self._rewind_mode_pending = False
        self._rewind_mode_active = False
        self._rewind_chunks_taken = 0
        self._rewind_credit_chunks_marked = 0
        self._rewind_selection_mode: Optional[str] = None
        self._rewind_to_takeover_pending = False

        # --- actor switch state ---
        self._actor_active = bool(args.start_with_actor) or not self.policy_switch_supported
        # Policy the chunks in rewind_history were run with; rewinding never
        # crosses a switch, because VLA chunks are not in replay.
        self._history_policy: Optional[str] = None

        # --- takeover state ---
        self._takeover_pending = False
        self._takeover_active = False
        self._control_generation = 0
        self._takeover_stop = threading.Event()
        self._takeover_input_done = threading.Event()
        self._takeover_return_requested = threading.Event()
        self._takeover_thread: Optional[threading.Thread] = None
        self._takeover_rl_thread: Optional[threading.Thread] = None
        self._chunk_buffer = InterventionChunkBuffer(self.chunk_length)
        self._chunk_buffer_lock = threading.Lock()
        self._chunk_ready = threading.Event()

        # --- episode accounting ---
        self._episode_idx = 0
        self._episode_reward = 0.0
        self._episode_chunks = 0
        self._episode_steps = 0
        self._episode_interventions = 0
        self._episode_intervention_actions = 0
        self._episode_success = False
        self._episode_done = False
        self._eval_episode_warned = False
        self._server_stopped = False
        self._episode_ready = threading.Event()
        self._main_transition_idle = threading.Event()
        self._main_transition_idle.set()

        self._stop_event = threading.Event()
        self._connection_error: Optional[Exception] = None
        self._button_thread = threading.Thread(
            target=self._button_loop, name="dobot-leader-button-monitor", daemon=True
        )
        self._button_thread.start()
        print(
            "[Takeover V7 sampled-replay] 接管流程：b 停顿 → "
            "录制键控制/松键保持 → 再次按 b 归还 policy；无需按 i"
        )

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._episode_ready.clear()
        self._stop_event.set()
        self._stop_takeover_thread()
        self._button_thread.join(timeout=5.0)
        self.keyboard.stop()
        self.leader.set_torque(2, True)

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

    # ------------------------------------------------------------------
    # observation
    # ------------------------------------------------------------------

    def _read_observation(self) -> dict[str, Any]:
        images = get_current_images()
        state = np.asarray(self.env.get_obs()["joint_positions"], dtype=np.float32).copy()
        if self.last_action is not None:
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

    def observation(self) -> dict[str, Any]:
        with self._env_lock:
            return self._read_observation()

    def safe_observation(self) -> dict[str, Any]:
        try:
            return self.observation()
        except Exception:
            if self.last_observation is None:
                raise
            return self.last_observation

    # ------------------------------------------------------------------
    # leader / button A
    # ------------------------------------------------------------------

    def _update_button_a(self, keys: np.ndarray, intervening: np.ndarray) -> None:
        """Toggle each leader arm's recovery unlock on a short A-button press."""
        pressed = np.asarray(keys[:, 0] == 0, dtype=bool)
        now = time.monotonic()
        with self._leader_state_lock:
            for side in range(2):
                if pressed[side] and not self._button_a_pressed[side]:
                    self._button_a_pressed_at[side] = now
                elif not pressed[side] and self._button_a_pressed[side]:
                    duration = now - self._button_a_pressed_at[side]
                    if duration < self.BUTTON_A_SHORT_PRESS_SECONDS:
                        self.leader_unlocked[side] = not self.leader_unlocked[side]
                        # Recording-button intervention also requires an unlocked
                        # leader, so locking A must not fight an active takeover.
                        should_unlock = self.leader_unlocked[side] or intervening[side]
                        self.leader.set_torque(side, not should_unlock)
                        name = "左臂" if side == 0 else "右臂"
                        if self.leader_unlocked[side]:
                            state = "解锁，可恢复初始位置"
                        elif intervening[side]:
                            state = "恢复模式关闭；人工接管结束后锁定"
                        else:
                            state = "锁定"
                        print(f"{name} A 键恢复模式：{state}（短按 {duration:.2f}s）")
                    self._button_a_pressed_at[side] = 0.0
            self._button_a_pressed = pressed

    def _button_loop(self) -> None:
        """Monitor recovery and recording buttons for the client lifetime."""
        interval = 1.0 / max(self.args.control_hz, 20.0)
        while not self._stop_event.is_set():
            try:
                keys = self.leader.get_keys()
                intervening = np.asarray(keys[:, 1] == 0, dtype=bool)
                self._update_button_a(keys, intervening)
                if (
                    self.args.takeover_enabled
                    and intervening.any()
                    and self._episode_ready.is_set()
                    and not self._episode_done
                    and self._rewind_mode_active
                    and not self._takeover_active
                ):
                    print(
                        "[Takeover] b 停顿期间检测到录制键 → "
                        "退出倒车模式并自动接管（无需按 i）"
                    )
                    self._apply_rewind_exit_correction()
                    self._deactivate_rewind_mode()
                    self._activate_takeover()
            except Exception as exc:  # never let the monitor kill the run
                print(f"主手按键监听异常（已忽略）：{exc}")
            self._stop_event.wait(interval)

    # ------------------------------------------------------------------
    # action helpers
    # ------------------------------------------------------------------

    def prepare_action(self, proposed: np.ndarray) -> np.ndarray:
        action = np.asarray(proposed, dtype=np.float64).copy().reshape(-1)
        if action.shape != (ACTION_DIM,) or not np.isfinite(action).all():
            raise ValueError(
                f"Invalid action: shape={action.shape}, finite={np.isfinite(action).all()}"
            )
        for index in GRIPPER_INDICES:
            action[index] = np.clip(action[index], 0.0, 1.0)
        return action

    def _apply_action(self, action: np.ndarray) -> np.ndarray:
        """Send one command to the follower and remember it as the tip."""
        action = self.prepare_action(action)
        with self._env_lock:
            if not self.args.dry_run:
                self.env.step(action, np.array([1, 1]))
            self.last_action = action.copy()
        return action

    def _leader_target(self, base: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Map leader displacement onto ``base`` for the arms being held."""
        assert self.last_action is not None
        keys = self.leader.get_keys()
        intervening = np.asarray(keys[:, 1] == 0, dtype=bool)
        released = self.was_intervening & ~intervening
        leader_now = self.leader.act({})
        for side in range(2):
            arm = slice(side * 7, side * 7 + 7)
            name = "左臂" if side == 0 else "右臂"
            if intervening[side] and not self.was_intervening[side]:
                with self._leader_state_lock:
                    self.leader.set_torque(side, False)
                leader_now = self.leader.act({})
                self.leader_origin[arm] = leader_now[arm]
                self.follower_origin[arm] = self.last_action[arm]
                print(f"{name}人工接管")
            elif released[side]:
                with self._leader_state_lock:
                    self.leader.set_torque(side, not self.leader_unlocked[side])
                print(f"{name}松开录制键，保持当前位姿")

        target = np.asarray(base, dtype=np.float64).copy()
        for side in range(2):
            arm = slice(side * 7, side * 7 + 7)
            if intervening[side]:
                if self.args.relative_takeover:
                    target[arm] = (
                        self.follower_origin[arm] + leader_now[arm] - self.leader_origin[arm]
                    )
                else:
                    target[arm] = leader_now[arm]
            elif released[side]:
                target[arm] = self.last_action[arm]
        self.was_intervening = intervening.copy()
        return self.prepare_action(target), intervening

    # ------------------------------------------------------------------
    # rewind (倒车) -- the client-side port of the Franka control node
    # ------------------------------------------------------------------

    def _rewind_ready(self) -> bool:
        return bool(self.args.rewind_enabled)

    def _handle_rewind_mode_toggle(self, *, chunk_in_flight: bool) -> None:
        """``b``: arm, enter, or leave rewind mode."""
        if not self._rewind_ready():
            return
        if self._episode_done:
            print("[Rewind] 回合已结束；先按 n 开始下一回合")
            return
        if self._rewind_mode_active:
            self._apply_rewind_exit_correction()
            self._deactivate_rewind_mode()
            print("[Rewind] 取消倒车 → 回到 policy")
            return
        if self._rewind_mode_pending:
            self._rewind_mode_pending = False
            print("[Rewind] 已取消挂起的倒车请求，继续 policy")
            return
        if chunk_in_flight:
            self._rewind_mode_pending = True
            self._takeover_pending = False
            print(
                "[Rewind] 已预约 — 等待当前 chunk 与 transition 完成；"
                f"再按一次 {self.args.rewind_mode_key} 取消"
            )
            return
        self._activate_rewind_mode()

    def _activate_rewind_mode(self) -> None:
        if not self._rewind_ready() or self._rewind_mode_active:
            return
        # A partially executed chunk is still rewindable: commit it so `r`
        # reverse-plays exactly what the arm just did.
        self.rewind_history.commit()
        self._rewind_mode_active = True
        self._rewind_mode_pending = False
        self._rewind_to_takeover_pending = False
        self._rewind_chunks_taken = 0
        self._rewind_credit_chunks_marked = 0
        self._rewind_selection_mode = None
        available = self.rewind_history.available()
        print(
            f"[Rewind] mode=ON | 可回退 chunks={available} "
            f"(缓存上限={self.rewind_history.history_size}) | "
            f"{self.args.rewind_step_key}=倒车一个 chunk，"
            f"{self.args.rewind_credit_key}=只标坏不动，"
            f"{self.args.rewind_mode_key}=退出，{self.args.takeover_key}=从当前位姿接管"
        )

    def _deactivate_rewind_mode(self) -> None:
        self._rewind_mode_active = False
        self._rewind_mode_pending = False
        self._rewind_to_takeover_pending = False

    def _handle_rewind_step(self) -> None:
        """``r``: reverse-play one finished chunk."""
        if not self._rewind_ready():
            return
        if not self._rewind_mode_active:
            print(
                f"[Rewind] 忽略 {self.args.rewind_step_key}："
                f"请先按 {self.args.rewind_mode_key} 进入倒车模式"
            )
            return
        if self._rewind_selection_mode == "credit":
            print(
                f"[Rewind] 已在 {self.args.rewind_credit_key} 只标坏模式；"
                "不能与物理倒车混用"
            )
            return
        self._rewind_selection_mode = "physical"
        try:
            self._execute_one_rewind()
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            self.robot_faulted = True
            print(f"[Rewind] 倒车失败：{exc}")

    def _execute_one_rewind(self) -> None:
        """Rewind one whole action chunk by reverse-playing its commands in order.

        For chunk [a0..a15] (arm at a15), move a14 -> a13 -> ... -> a0 -> the
        previous chunk's tip, instead of jumping straight to the previous chunk
        end, which would be a single large unconstrained motion.
        """
        removed_chunk = self.rewind_history.pop_chunk()
        if removed_chunk is None:
            print("[Rewind] 没有可回退的 chunk（available=0）")
            return
        final_tip = self.rewind_history.peek_tip()
        if final_tip is None:
            print("[Rewind] 没有可落脚的 anchor，放弃本次倒车")
            return

        # Reverse through the chunk, then land on the previous chunk tip / anchor.
        # Skip the last entry of removed_chunk -- that is where the arm already is.
        waypoints = list(reversed(removed_chunk[:-1])) + [final_tip]
        tip_now = np.asarray(removed_chunk[-1], dtype=np.float64)
        span = float(np.max(np.abs(np.asarray(final_tip, dtype=np.float64) - tip_now)))
        remaining_after = self.rewind_history.available()
        print(
            f"[Rewind] 反向回放 chunk steps={len(removed_chunk)} → "
            f"{len(waypoints)} 次运动（最大关节跨度 {np.rad2deg(span):.1f}°）| "
            f"剩余 chunks={remaining_after}"
        )

        prev = tip_now.copy()
        for index, waypoint in enumerate(waypoints):
            if not self._rewind_mode_active:
                raise RuntimeError("倒车模式在回放过程中被取消")
            target = self.prepare_action(waypoint)
            step = float(np.max(np.abs(target - prev)))
            # Skip no-ops that only burn time.
            if step < 1e-5:
                continue
            if not self.args.dry_run:
                with self._env_lock:
                    move_linearly(
                        self.env,
                        prev,
                        target,
                        max_steps=self.args.rewind_max_steps_per_move,
                        max_step=self.args.rewind_max_step,
                    )
            with self._env_lock:
                self.last_action = target.copy()
            prev = target
            if (index + 1) % 4 == 0 or index + 1 == len(waypoints):
                print(
                    f"[Rewind] 反向进度 {index + 1}/{len(waypoints)} "
                    f"Δstep={np.rad2deg(step):.2f}°"
                )

        self._rewind_chunks_taken += 1
        print(
            f"[Rewind] ready | chunks_rewound={self._rewind_chunks_taken} "
            f"available={self.rewind_history.available()}"
        )

    def _handle_rewind_credit_step(self) -> None:
        """``q``: move the replay credit boundary back one chunk, no robot motion."""
        if not self._rewind_ready():
            return
        if not self._rewind_mode_active:
            print(
                f"[Rewind credit] 忽略 {self.args.rewind_credit_key}："
                f"请先按 {self.args.rewind_mode_key} 进入倒车模式"
            )
            return
        if self._rewind_selection_mode == "physical":
            print(
                f"[Rewind credit] 已在物理倒车（{self.args.rewind_step_key}）模式；"
                "不能与只标坏混用"
            )
            return
        if self._history_policy == "vla":
            print("[Rewind credit] 当前是 VLA 段，这些 chunk 不在 replay 中，无需标坏")
            return
        if self._rewind_credit_chunks_marked >= int(self._episode_chunks):
            print("[Rewind credit] 本回合没有更早的已存 chunk 可标记")
            return
        self._rewind_selection_mode = "credit"
        self._rewind_credit_chunks_marked += 1
        print(
            f"[Rewind credit] 已标记 bad_chunks={self._rewind_credit_chunks_marked}"
            "（只改 replay，机械臂不动）"
        )

    def _apply_rewind_exit_correction(self) -> None:
        """Apply either physical-rewind replacement or ``q`` credit isolation."""
        chunks = int(self._rewind_chunks_taken)
        credit_chunks = int(self._rewind_credit_chunks_marked)
        selection_mode = self._rewind_selection_mode
        self._rewind_chunks_taken = 0
        self._rewind_credit_chunks_marked = 0
        self._rewind_selection_mode = None

        if self._history_policy == "vla":
            if chunks or credit_chunks:
                print("[Rewind] 倒回的是 VLA 段（未写入 replay），不做 replay 修正")
            return
        if selection_mode == "credit" and credit_chunks > 0:
            reward = float(self.args.rewind_credit_exit_reward)
            request = {
                "rlt/request": "rewind_credit_correction",
                "terminal_reward": reward,
                "bad_chunks": credit_chunks,
                "prefix_reward": float(self.args.rewind_prefix_reward),
            }
        else:
            reward = float(self.args.rewind_physical_exit_reward)
            request = {
                "rlt/request": "rewind_exit_correction",
                "terminal_reward": reward,
                "chunks_rewound": chunks,
            }
        if reward == 0.0:
            return
        try:
            with self._policy_lock:
                response = self.policy.infer(request)
            if response.get("applied"):
                self._episode_reward += float(response.get("episode_reward_delta", 0.0))
            if request["rlt/request"] == "rewind_credit_correction":
                print(
                    f"[Rewind credit] applied={response.get('applied')} "
                    f"bad_chunks={response.get('bad_chunks')} "
                    f"prefix={response.get('prefix_index')} "
                    f"bad_start={response.get('bad_branch_start_index')} "
                    f"bad_terminal={response.get('bad_terminal_index')} "
                    f"terminal_reward={reward:.3f} "
                    f"prefix_reward={float(self.args.rewind_prefix_reward):.3f}"
                )
            else:
                print(
                    f"[Rewind] exit correction applied={response.get('applied')} "
                    f"reward={reward:.3f} chunks_rewound={chunks} "
                    f"penalized={response.get('penalized_index')} "
                    f"bad_start={response.get('bad_branch_start_index')} "
                    f"bad_terminal={response.get('bad_terminal_index')} "
                    f"replacement_anchor={response.get('replacement_anchor_index')}"
                )
        except Exception as exc:
            print(f"[Rewind] exit correction 失败：{exc}")

    # ------------------------------------------------------------------
    # takeover (接管)
    # ------------------------------------------------------------------

    def _handle_takeover_toggle(self, *, chunk_in_flight: bool) -> None:
        """``i``: request, enter, or leave takeover."""
        if not self.args.takeover_enabled:
            return
        if self._takeover_active:
            self._deactivate_takeover()
            return
        if self._episode_done:
            print("[Takeover] 回合已结束；先按 n 开始下一回合")
            return
        if self._rewind_mode_active:
            # Leaving rewind through i keeps the exit correction semantics.
            self._apply_rewind_exit_correction()
            self._deactivate_rewind_mode()
            self._activate_takeover()
            return
        if self._takeover_pending:
            self._takeover_pending = False
            print("[Takeover] 已取消挂起的接管请求")
            return
        if chunk_in_flight:
            self._takeover_pending = True
            self._rewind_mode_pending = False
            print(
                "[Takeover] 已预约 — 等待当前 chunk 与 transition 完成；"
                f"再按一次 {self.args.takeover_key} 取消"
            )
            return
        self._activate_takeover()

    def _activate_takeover(self) -> None:
        # Serialize the ownership handoff with follower commands.  If policy is
        # in the middle of a servo tick, that tick completes first; no later
        # policy tick can pass the active check in _execute_chunk.
        with self._env_lock:
            if self._takeover_active:
                return
            self._takeover_pending = False
            self._takeover_active = True
            self._control_generation += 1
        self._takeover_stop.clear()
        self._takeover_input_done.clear()
        self._takeover_return_requested.clear()
        self._chunk_ready.clear()
        with self._chunk_buffer_lock:
            self._chunk_buffer.clear()
        # A takeover chunk must not be mixed into the policy rewind history.
        self.rewind_history.drop_building()
        # Teleop and the act/transition round trip run on separate threads so a
        # slow server never stutters the arm the human is holding.
        self._takeover_thread = threading.Thread(
            target=self._takeover_loop, name="dobot-takeover", daemon=True
        )
        self._takeover_rl_thread = threading.Thread(
            target=self._takeover_rl_loop, name="dobot-takeover-rl", daemon=True
        )
        self._takeover_thread.start()
        self._takeover_rl_thread.start()
        print(
            "[Takeover] 录制键接管：遥操 "
            f"{self.args.control_hz:g} Hz，数据采样 "
            f"{self.args.takeover_record_hz:g} Hz；松开后保持当前位姿；"
            f"按 {self.args.rewind_mode_key} 归还 policy；"
            f"单条 transition 最多 {self.chunk_length} 个真实 step"
        )

    def _request_takeover_return(self) -> None:
        """Ask teleop to close its current chunk, then return to policy."""
        if not self._takeover_active or self._takeover_return_requested.is_set():
            return
        self._takeover_return_requested.set()
        print("[Takeover] 已请求归还 policy；正在收尾人工 transition")

    def _deactivate_takeover(self) -> None:
        was_active = self._takeover_active
        self._takeover_active = False
        self._takeover_pending = False
        self._stop_takeover_thread()
        self._takeover_return_requested.clear()
        with self._leader_state_lock:
            self.leader.set_torque(2, True)
            self.was_intervening[:] = False
        with self._chunk_buffer_lock:
            self._chunk_buffer.clear()
        if was_active:
            print("[Takeover] 接管结束 → 回到 policy")

    def _stop_takeover_thread(self) -> None:
        self._takeover_stop.set()
        self._takeover_input_done.set()
        self._chunk_ready.set()
        current = threading.current_thread()
        for attribute in ("_takeover_thread", "_takeover_rl_thread"):
            thread = getattr(self, attribute)
            # Never join the thread we are running on (the RL loop ends the
            # episode itself when the operator presses s/f during takeover).
            if thread is not None and thread.is_alive() and thread is not current:
                thread.join(timeout=5.0)
            # Retain a timed-out worker reference so reset_episode can wait for
            # episode_end to finish instead of letting it mutate a new episode.
            if thread is None or thread is current or not thread.is_alive():
                if getattr(self, attribute) is thread:
                    setattr(self, attribute, None)

    def _takeover_loop(self) -> None:
        """Drive at control_hz and sample replay actions at takeover_record_hz."""
        interval = 1.0 / self.args.control_hz
        record_interval = 1.0 / self.args.takeover_record_hz
        deadline = time.monotonic()
        next_record_at = deadline
        saw_recording = False
        while not self._takeover_stop.is_set() and not self._stop_event.is_set():
            try:
                base = self.last_action
                if base is None:
                    break
                target, intervening = self._leader_target(base)
                recording = bool(intervening.any())

                if self._takeover_return_requested.is_set():
                    with self._chunk_buffer_lock:
                        action_count = self._chunk_buffer.action_count
                    if action_count:
                        boundary_obs = self.observation()
                        with self._chunk_buffer_lock:
                            chunk = self._chunk_buffer.close_on_release(boundary_obs)
                        self._chunk_ready.set()
                        print(
                            "[Takeover] b 归还：提交最后人工片段 "
                            f"steps_executed={chunk.steps_executed}"
                        )
                    self._takeover_input_done.set()
                    self._chunk_ready.set()
                    return

                if not recording:
                    if saw_recording:
                        with self._chunk_buffer_lock:
                            action_count = self._chunk_buffer.action_count
                        if action_count:
                            boundary_obs = self.observation()
                            with self._chunk_buffer_lock:
                                chunk = self._chunk_buffer.close_on_release(boundary_obs)
                            self._chunk_ready.set()
                            print(
                                "[Takeover] 录制键松开：提交 "
                                f"steps_executed={chunk.steps_executed}；"
                                "保持当前位姿，接管模式继续"
                            )
                        saw_recording = False
                    deadline += interval
                    self._takeover_stop.wait(max(0.0, deadline - time.monotonic()))
                    continue

                now = time.monotonic()
                if not saw_recording:
                    # Record the first command immediately on every new press.
                    next_record_at = now
                saw_recording = True
                record_this_action = now >= next_record_at
                step_obs = None
                if record_this_action:
                    with self._chunk_buffer_lock:
                        needs_start = self._chunk_buffer.needs_start_observation
                        needs_boundary = self._chunk_buffer.needs_boundary_observation
                    if needs_start:
                        start_obs = self.observation()
                        with self._chunk_buffer_lock:
                            self._chunk_buffer.start(start_obs)
                    elif needs_boundary:
                        boundary_obs = self.observation()
                        with self._chunk_buffer_lock:
                            self._chunk_buffer.close_at_boundary(boundary_obs)
                        self._chunk_ready.set()
                    with self._chunk_buffer_lock:
                        offset = self._chunk_buffer.action_count
                    step_obs = (
                        self.observation()
                        if self.takeover_step_obs and offset in self.step_obs_offsets
                        else None
                    )
                action = self._apply_action(target)
                if record_this_action:
                    with self._chunk_buffer_lock:
                        self._chunk_buffer.append_action(action, step_obs)
                        if self._chunk_buffer.completed_count:
                            self._chunk_ready.set()
                    next_record_at += record_interval
                    if next_record_at <= now:
                        # Do not duplicate one physical command to catch up
                        # after a delayed control tick.
                        next_record_at = now + record_interval
            except KeyboardInterrupt:
                break
            except Exception as exc:
                print(f"[Takeover] 控制异常，接管终止：{exc}")
                self.robot_faulted = True
                self._takeover_active = False
                self._takeover_input_done.set()
                self._chunk_ready.set()
                break
            deadline += interval
            self._takeover_stop.wait(max(0.0, deadline - time.monotonic()))
            if time.monotonic() - deadline > interval:
                deadline = time.monotonic()

    def _takeover_rl_loop(self) -> None:
        """Ship each completed takeover chunk as its own act/transition pair.

        Runs off the teleop thread on purpose: the human keeps moving the arm at
        control_hz while replay samples arrive at takeover_record_hz and this
        waits on the policy server.
        """
        while not self._takeover_stop.is_set() and not self._stop_event.is_set():
            self._chunk_ready.wait(timeout=0.2)
            if self._takeover_stop.is_set() or self._stop_event.is_set():
                return
            with self._chunk_buffer_lock:
                chunk = self._chunk_buffer.pop_completed()
                if chunk is None or self._chunk_buffer.completed_count == 0:
                    self._chunk_ready.clear()
            if chunk is None:
                if self._takeover_input_done.is_set():
                    self._takeover_active = False
                    self._takeover_stop.set()
                    with self._leader_state_lock:
                        self.leader.set_torque(2, True)
                        self.was_intervening[:] = False
                    print("[Takeover] b 收尾完成 → 归还 policy")
                    return
                continue
            try:
                # A policy act/transition pair may already be in flight when the
                # hardware button starts takeover.  Let the main thread discard
                # or store it before requesting an intervention transition id.
                while not self._main_transition_idle.wait(timeout=0.1):
                    if self._takeover_stop.is_set() or self._stop_event.is_set():
                        return
                result = self._act(chunk.observation)
                if result is None:
                    self._takeover_stop.set()
                    return
                reward_signal, small_presses = self.keyboard.take_reward()
                if reward_signal == "next":
                    reward_signal = None
                rewards, done, bootstrap_mask, info = self._build_rewards(
                    reward_signal,
                    steps_executed=chunk.steps_executed,
                    small_progress_presses=small_presses,
                )
                info["intervention"] = True
                info["takeover_continuous"] = (
                    chunk.steps_executed == self.chunk_length
                )
                info["recording_button_takeover"] = True
                info["takeover_control_hz"] = float(self.args.control_hz)
                info["takeover_record_hz"] = float(self.args.takeover_record_hz)
                self._transition(
                    transition_id=str(result["transition_id"]),
                    next_observation=chunk.next_observation,
                    rewards=rewards,
                    done=done,
                    bootstrap_mask=bootstrap_mask,
                    info=info,
                    intervention=True,
                    step_observations=list(chunk.step_observations),
                    action_chunk=chunk.actions,
                    action_chunk_space="robot",
                    steps_executed=chunk.steps_executed,
                )
                self._episode_interventions += 1
                self._episode_intervention_actions += int(chunk.steps_executed)
                if done:
                    self._episode_success = bool(info.get("success", False))
                    self._finish_episode()
                    self._takeover_active = False
                    self._takeover_stop.set()
                    return
            except Exception as exc:
                if isinstance(exc, NETWORK_ERRORS):
                    # Hand reconnect back to the main thread; stop accumulating
                    # human chunks against a dead websocket.
                    self._connection_error = exc
                    self._takeover_stop.set()
                    self._takeover_active = False
                    return
                # Keep teleop alive: the operator still has the arm, and the
                # next chunk may well go through.
                print(f"[Takeover] act/transition 失败（本 chunk 丢弃）：{exc}")
                time.sleep(0.05)

    # ------------------------------------------------------------------
    # RLT protocol
    # ------------------------------------------------------------------

    def _act(self, observation: dict[str, Any]) -> Optional[dict[str, Any]]:
        request: dict[str, Any] = {"rlt/request": "act", "observation": observation}
        if self.policy_switch_supported:
            request["policy"] = self._current_policy()
        if self.args.exploration_noise_sigma >= 0.0:
            request["exploration_noise_sigma"] = float(self.args.exploration_noise_sigma)
        with self._policy_lock:
            result = self.policy.infer(request)
        if result.get("should_stop", False):
            print("[RLT] 服务端请求停止")
            self._server_stopped = True
            return None
        if bool(result.get("is_eval_episode")) or str(result.get("mode", "")) == "eval":
            if not self._eval_episode_warned:
                self._eval_episode_warned = True
                print(
                    "[RLT EVAL] noise=0 自主评估集 — 请勿接管/勿按 "
                    f"{self.args.takeover_key}；仅用 s/f 标成功失败 | "
                    f"train_episodes={result.get('total_episodes')} "
                    f"eval_episodes={result.get('total_eval_episodes')}"
                )
        return result

    def _build_rewards(
        self,
        reward_signal: Optional[str],
        *,
        steps_executed: int,
        small_progress_presses: int = 0,
        fault_reason: Optional[str] = None,
    ) -> tuple[np.ndarray, bool, float, dict[str, Any]]:
        rewards = np.zeros(self.chunk_length, dtype=np.float32)
        reward_index = max(0, min(self.chunk_length, max(steps_executed, 1)) - 1)
        done = False
        bootstrap_mask = 1.0
        info: dict[str, Any] = {
            "steps_executed": int(steps_executed),
            "client": "efficient_rl_stage2_client",
        }
        if reward_signal == "progress":
            rewards[reward_index] = float(self.args.progress_reward)
            info["progress"] = True
        elif reward_signal == "progress_small":
            rewards[reward_index] = float(self.args.small_progress_reward) * max(
                1, int(small_progress_presses)
            )
            info["progress"] = True
            info["progress_small"] = True
            info["progress_small_presses"] = max(1, int(small_progress_presses))
        elif reward_signal == "regress":
            rewards[reward_index] = float(self.args.regress_reward)
            info["regress"] = True
        elif reward_signal == "success":
            rewards[reward_index] = float(self.args.success_reward)
            done = True
            info["success"] = True
        elif reward_signal == "failure":
            rewards[reward_index] = float(self.args.failure_reward)
            done = True
            info["success"] = False
        elif reward_signal == "fault":
            rewards[reward_index] = float(self.args.fault_terminal_reward)
            done = True
            bootstrap_mask = 0.0
            info["success"] = False
            info["fault"] = True
            info["fault_reason"] = str(fault_reason or "dobot_fault")
        if (
            self.args.max_episode_chunks > 0
            and self._episode_chunks + 1 >= self.args.max_episode_chunks
            and not done
        ):
            done = True
            info["success"] = False
            info["timeout"] = True
        return rewards, done, bootstrap_mask, info

    def _transition(
        self,
        *,
        transition_id: str,
        next_observation: dict[str, Any],
        rewards: np.ndarray,
        done: bool,
        bootstrap_mask: float,
        info: dict[str, Any],
        intervention: bool,
        action_chunk: np.ndarray,
        action_chunk_space: str,
        steps_executed: int,
        step_observations: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        with self._policy_lock:
            response = self.policy.infer(
                {
                    "rlt/request": "transition",
                    "transition_id": transition_id,
                    "next_observation": next_observation,
                    "rewards": rewards,
                    "done": bool(done),
                    "bootstrap_mask": float(bootstrap_mask),
                    "info": info,
                    "intervention": bool(intervention),
                    "action_chunk": action_chunk,
                    "action_chunk_space": action_chunk_space,
                    "step_observations": step_observations or [],
                }
            )
        self._episode_chunks += 1
        self._episode_steps += int(steps_executed)
        self._episode_reward += float(rewards.sum())
        if done:
            self._episode_done = True
            self._episode_success = bool(info.get("success", False))
        print(
            f"[RLT transition] stored={response.get('stored')} mode={response.get('mode')} "
            f"steps={steps_executed} reward={float(rewards.sum()):.2f} done={done} "
            f"bootstrap_mask={bootstrap_mask:.1f} intervention={intervention} "
            f"buffer={response.get('buffer_size')} updates={response.get('total_updates')}"
        )

    def _discard(self, transition_id: str, reason: str) -> None:
        try:
            with self._policy_lock:
                self.policy.infer(
                    {"rlt/request": "discard", "transition_id": transition_id, "reason": reason}
                )
            print(f"[RLT] 已丢弃 transition {transition_id}（{reason}）")
        except Exception as exc:
            print(f"[RLT] 丢弃 transition 失败：{exc}")

    def _finish_episode(self) -> None:
        stats = {
            "episode_reward": float(self._episode_reward),
            "episode_chunks": int(self._episode_chunks),
            "episode_steps": int(self._episode_steps),
            "episode_interventions": int(self._episode_interventions),
            "episode_intervention_actions": int(self._episode_intervention_actions),
            "success": bool(self._episode_success),
        }
        try:
            with self._policy_lock:
                response = self.policy.infer({"rlt/request": "episode_end", "stats": stats})
            print(
                f"[RLT episode {self._episode_idx}] success={stats['success']} "
                f"reward={stats['episode_reward']:.2f} chunks={stats['episode_chunks']} "
                f"interventions={stats['episode_interventions']} "
                f"intervention_actions={stats['episode_intervention_actions']} "
                f"buffer={response.get('buffer_size')} updates={response.get('total_updates')} "
                f"checkpoint={response.get('checkpoint')}"
            )
        except Exception as exc:
            print(f"[RLT episode_end] 失败：{exc}")
        finally:
            self._episode_ready.clear()
            self._episode_idx += 1
            self._episode_reward = 0.0
            self._episode_chunks = 0
            self._episode_steps = 0
            self._episode_success = False
            self._episode_interventions = 0
            self._episode_intervention_actions = 0
            self._eval_episode_warned = False
            self._episode_done = True

    # ------------------------------------------------------------------
    # chunk execution
    # ------------------------------------------------------------------

    def _select_actions(self, actions: np.ndarray) -> np.ndarray:
        max_actions = int(self.args.max_actions_to_publish)
        if max_actions <= 0 or len(actions) <= max_actions:
            return actions
        if self.args.use_last_actions:
            return actions[-max_actions:]
        return actions[:max_actions]

    def _execute_chunk(self, actions: np.ndarray) -> int:
        """Send one chunk to the follower, recording it for rewind as we go.

        Keyboard mode keys are applied at a transition boundary. Hardware
        recording buttons activate the independent takeover loop; this method
        stops before publishing another stale policy command once ownership
        changes.
        """
        interval = 1.0 / self.args.control_hz
        deadline = time.monotonic()
        if self.last_action is not None:
            self.rewind_history.seed(self.last_action)
        executed = 0
        self._executed_actions = []
        self._executed_step_observations = []
        self._executed_intervention_masks = []
        for index, proposed in enumerate(actions):
            self.keyboard.poll()
            self._drain_edges(chunk_in_flight=True)
            with self._env_lock:
                if self._takeover_active:
                    break
                if index in self.step_obs_offsets:
                    self._executed_step_observations.append(
                        {"offset": index, "observation": self.observation()}
                    )
                action = self._apply_action(proposed)
                self._executed_actions.append(
                    np.asarray(action, dtype=np.float32).copy()
                )
                self._executed_intervention_masks.append(
                    np.zeros(2, dtype=bool)
                )
                self.rewind_history.record(action)
                executed += 1
            if self.args.action_publish_interval > 0 and index < len(actions) - 1:
                time.sleep(self.args.action_publish_interval)
            deadline += interval
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            else:
                deadline = time.monotonic()
        self.rewind_history.commit()
        return executed

    def _drain_edges(self, *, chunk_in_flight: bool) -> None:
        """Dispatch queued b/r/q/i presses."""
        while True:
            edge = self.keyboard.take_edge()
            if edge is None:
                return
            if edge == "rewind_mode":
                if self._takeover_active:
                    self._request_takeover_return()
                else:
                    self._handle_rewind_mode_toggle(chunk_in_flight=chunk_in_flight)
            elif edge == "rewind_step":
                if chunk_in_flight:
                    print(
                        f"[Rewind] 当前 chunk 未执行完，忽略 {self.args.rewind_step_key}；"
                        f"先按 {self.args.rewind_mode_key} 预约倒车"
                    )
                else:
                    self._handle_rewind_step()
            elif edge == "rewind_credit":
                if chunk_in_flight:
                    print(
                        f"[Rewind credit] 当前 chunk 未执行完，忽略 "
                        f"{self.args.rewind_credit_key}"
                    )
                else:
                    self._handle_rewind_credit_step()
            elif edge == "takeover":
                self._handle_takeover_toggle(chunk_in_flight=chunk_in_flight)
            elif edge == "actor_switch":
                self._toggle_actor(chunk_in_flight=chunk_in_flight)

    def _current_policy(self) -> str:
        return "actor" if self._actor_active else "vla"

    def _toggle_actor(self, *, chunk_in_flight: bool) -> None:
        """``a``: switch VLA <-> actor; the next act request uses the new policy."""
        if self._episode_done:
            print("[Actor switch] 回合已结束；先按 n 开始下一回合")
            return
        self._actor_active = not self._actor_active
        when = "当前 chunk 执行完后" if chunk_in_flight else "下一个 chunk"
        if self._actor_active:
            print(f"[Actor switch] → actor（{when}生效）：actor chunk 与期间的接管写入 replay")
        else:
            print(f"[Actor switch] → VLA（{when}生效）：VLA chunk 与期间的接管只执行，不写 replay")

    # ------------------------------------------------------------------
    # episode driving
    # ------------------------------------------------------------------

    def _print_episode_banner(self) -> None:
        keys = [
            "s/空格=成功",
            "f=失败",
            "p=进展",
            "o=小进展(累加)",
            "x=倒退",
            "n=下一回合",
        ]
        if self.args.rewind_enabled:
            keys += [
                f"{self.args.rewind_mode_key}=进入/退出倒车",
                f"{self.args.rewind_step_key}=倒车一个 chunk",
                f"{self.args.rewind_credit_key}=只标坏不动",
            ]
        if self.args.takeover_enabled:
            keys.append("b 后录制键=接管/松开=保持/再按 b=归还")
            keys.append(f"{self.args.takeover_key}=可选键盘接管快捷键")
        if self.policy_switch_supported:
            keys.append(f"{self.args.actor_switch_key}=切换 actor/VLA")
        keys.append("Ctrl+C=停止")
        print(f"回合 {self._episode_idx} 开始：" + "，".join(keys))
        if self.policy_switch_supported:
            print(
                f"[Actor switch] 本回合从 {self._current_policy()} 开始；"
                "只有 actor 段（含期间接管）写入 replay"
            )

    def reset_episode(self) -> dict[str, Any]:
        self._episode_ready.clear()
        self._main_transition_idle.set()
        self.keyboard.stop()
        # Do this unconditionally.  A terminal s/f handled by the takeover RL
        # thread sets _takeover_active=False before it exits, so an
        # active-only cleanup would leave the previous episode's thread/events
        # alive and let them leak into the next episode.
        self._deactivate_takeover()
        lingering_workers = [
            thread.name
            for thread in (self._takeover_thread, self._takeover_rl_thread)
            if thread is not None and thread.is_alive()
        ]
        if lingering_workers:
            print(
                "[Episode reset] 等待上一回合完成 episode_end："
                + ", ".join(lingering_workers)
            )
            for attribute in ("_takeover_thread", "_takeover_rl_thread"):
                thread = getattr(self, attribute)
                if thread is not None and thread.is_alive():
                    thread.join()
                if getattr(self, attribute) is thread:
                    setattr(self, attribute, None)
        self._deactivate_rewind_mode()
        self._rewind_chunks_taken = 0
        self._rewind_credit_chunks_marked = 0
        self._rewind_selection_mode = None
        self.rewind_history.clear()
        self._history_policy = None
        self._actor_active = bool(self.args.start_with_actor) or not self.policy_switch_supported
        # Clear keyboard state only after old episode workers have stopped, so
        # no old worker can consume or re-expose an s/b event after this point.
        self.keyboard.clear()
        self._takeover_stop.clear()
        self._takeover_input_done.clear()
        self._takeover_return_requested.clear()
        self._chunk_ready.clear()
        self._episode_done = False
        print("[Episode reset] 上一回合的按键、接管、倒车状态已全部清除")
        with self._leader_state_lock:
            self.leader.set_torque(2, True)
            self.was_intervening[:] = False
            self.leader_unlocked[:] = False
            self._button_a_pressed[:] = False
            self._button_a_pressed_at[:] = 0.0
        self.leader_origin[:] = 0.0
        self.follower_origin[:] = 0.0

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
                print(f"复位失败，训练保持暂停：{exc}")
        input("请布置下一回合场景，确认安全后按 Enter：")

        obs = self.env.get_obs()
        self.last_action = np.asarray(obs["joint_positions"], dtype=np.float64).copy()
        for index in GRIPPER_INDICES:
            self.last_action[index] = 1.0
        observation = self.observation()
        self.keyboard.start()
        self._episode_ready.set()
        self._print_episode_banner()
        print(
            "人工接管：先按 b 进入停顿，再按住任一主手录制键移动；"
            "松开录制键只保持当前位置，再按 b 才归还 policy，无需按 i；"
            "短按主手 A 键可解锁/锁定对应主臂以恢复初始位置"
        )
        return observation

    def _wait_in_rewind_mode(self) -> None:
        """Block policy inference while the operator rewinds / marks chunks."""
        while self._rewind_mode_active and not self._stop_event.is_set():
            self.keyboard.poll()
            # Reward keys are meaningless with no transition in flight.
            if self.keyboard.peek_reward() is not None:
                ignored, _ = self.keyboard.take_reward()
                print(f"[Rewind] 倒车模式下忽略奖励键：{ignored}")
            self._drain_edges(chunk_in_flight=False)
            if self._takeover_active:
                return
            time.sleep(0.02)

    def _wait_in_takeover(self) -> None:
        while self._takeover_active and not self._stop_event.is_set():
            self.keyboard.poll()
            self._drain_edges(chunk_in_flight=False)
            if self._episode_done:
                return
            time.sleep(0.02)

    def _pause_for_next_episode(self) -> None:
        print("[RLT] 回合结束；请复位场景，按 n 开始下一回合")
        while not self._stop_event.is_set():
            self.keyboard.poll()
            if self.keyboard.take_reward()[0] == "next":
                return
            # Drop rewind/takeover edges queued after the episode ended.
            while self.keyboard.take_edge() is not None:
                pass
            time.sleep(0.05)

    def run_episode(self) -> None:
        observation = self.reset_episode()
        while not self._stop_event.is_set() and not self._server_stopped:
            if self._connection_error is not None:
                raise self._connection_error
            self.keyboard.poll()
            self._drain_edges(chunk_in_flight=False)

            if self._rewind_mode_active:
                self._wait_in_rewind_mode()
                observation = self.safe_observation()
                continue
            if self._takeover_active:
                self._wait_in_takeover()
                if self._episode_done:
                    break
                observation = self.safe_observation()
                continue

            # Close the race between the active check above and an automatic
            # hardware-button takeover starting from the monitor thread.
            policy = self._current_policy()
            if policy != self._history_policy:
                if self._history_policy is not None and self.rewind_history.available():
                    print("[Actor switch] 倒车缓存已清空：倒车不跨越 actor/VLA 切换点")
                self.rewind_history.clear()
                self._history_policy = policy

            with self._env_lock:
                if self._takeover_active:
                    continue
                act_generation = self._control_generation
                self._main_transition_idle.clear()
            result = self._act(observation)
            if result is None:
                self._main_transition_idle.set()
                break
            # A takeover armed while act() was in flight must not execute a chunk
            # inferred from a state the human has since changed.
            if (
                self._takeover_active
                or self._control_generation != act_generation
            ):
                self._discard(str(result["transition_id"]), "takeover_during_infer")
                self._main_transition_idle.set()
                observation = self.safe_observation()
                continue

            actions_raw = np.asarray(result["actions"], dtype=np.float32)
            normalized_raw = result.get("normalized_actions")
            if normalized_raw is None:
                if self.args.require_normalized_replay and self.replay_action_space == "normalized":
                    raise RuntimeError(
                        "RLT act response is missing normalized_actions; "
                        "refusing to collect a mixed-space transition"
                    )
                replay_actions = actions_raw
                replay_space = "robot"
            else:
                normalized_actions = np.asarray(normalized_raw, dtype=np.float32)
                if normalized_actions.shape != actions_raw.shape:
                    raise ValueError(
                        "normalized_actions shape does not match robot actions: "
                        f"{normalized_actions.shape} vs {actions_raw.shape}"
                    )
                replay_actions = normalized_actions
                replay_space = "normalized"

            actions_to_publish = self._select_actions(actions_raw)
            fault_reason: Optional[str] = None
            try:
                steps_executed = self._execute_chunk(actions_to_publish)
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                self.robot_faulted = True
                self.rewind_history.drop_building()
                fault_reason = f"{type(exc).__name__}: {exc}"
                steps_executed = len(getattr(self, "_executed_actions", []))
                print(f"机械臂故障，本回合以 fault 终止：{fault_reason}")

            # Snapshot the policy boundary atomically with the ownership
            # generation. A takeover beginning after this snapshot belongs to
            # the next transition and may safely run during the network RPC.
            with self._env_lock:
                takeover_invalidated_policy = (
                    self._control_generation != act_generation
                )
                next_observation = self.safe_observation()
            if takeover_invalidated_policy:
                self._discard(
                    str(result["transition_id"]),
                    "takeover_during_policy_execution",
                )
                self._main_transition_idle.set()
                observation = next_observation
                continue
            small_presses = 0
            if fault_reason is not None:
                reward_signal = "fault"
            else:
                reward_signal, small_presses = self.keyboard.take_reward()
                if reward_signal == "next":
                    reward_signal = None
            rewards, done, bootstrap_mask, info = self._build_rewards(
                reward_signal,
                steps_executed=max(steps_executed, 1),
                small_progress_presses=small_presses,
                fault_reason=fault_reason,
            )
            intervention_mask = np.zeros((self.chunk_length, 2), dtype=bool)
            executed_masks = getattr(self, "_executed_intervention_masks", [])
            if executed_masks:
                intervention_mask[:len(executed_masks)] = np.stack(executed_masks)
            intervention_occurred = bool(
                intervention_mask[:steps_executed].any()
            )
            info["intervention"] = intervention_occurred
            info["intervention_occurred"] = intervention_occurred
            info["intervention_mask"] = intervention_mask
            info["steps_executed"] = steps_executed
            executed_rows = getattr(self, "_executed_actions", [])
            replay_actions = np.asarray(actions_to_publish, dtype=np.float32).copy()
            if executed_rows:
                replay_actions[:len(executed_rows)] = np.stack(executed_rows)
                replay_actions[len(executed_rows):] = executed_rows[-1]
            self._transition(
                transition_id=str(result["transition_id"]),
                next_observation=next_observation,
                rewards=rewards,
                done=done,
                bootstrap_mask=bootstrap_mask,
                info=info,
                intervention=intervention_occurred,
                action_chunk=replay_actions,
                action_chunk_space="robot",
                step_observations=getattr(self, "_executed_step_observations", []),
                steps_executed=max(steps_executed, 1),
            )
            self._main_transition_idle.set()
            if intervention_occurred:
                self._episode_interventions += 1
                self._episode_intervention_actions += int(
                    intervention_mask[:steps_executed].any(axis=1).sum()
                )
            observation = next_observation

            if done:
                self._finish_episode()
                break

            # Chunk finished and transition stored -- now honour a pending b/i.
            if self._rewind_mode_pending:
                print("[Rewind] 当前 chunk 完成、transition 已存 — 进入倒车模式")
                self._activate_rewind_mode()
                continue
            if self._takeover_pending:
                print("[Takeover] 当前 chunk 完成、transition 已存 — 进入接管")
                self._activate_takeover()
                continue

        if (
            self.args.pause_after_episode
            and not self._stop_event.is_set()
            and not self._server_stopped
        ):
            self._pause_for_next_episode()

    def run(self) -> None:
        while not self._stop_event.is_set() and not self._server_stopped:
            self.run_episode()
        if self._server_stopped:
            print("[RLT] 服务端已结束训练，客户端退出")


def main(args: Args) -> int:
    if args.control_hz <= 0:
        raise ValueError("control_hz must be positive")
    if args.takeover_record_hz <= 0:
        raise ValueError("takeover_record_hz must be positive")
    if args.takeover_record_hz > args.control_hz:
        raise ValueError("takeover_record_hz must not exceed control_hz")
    if args.rewind_enabled:
        edge_keys = {
            args.rewind_mode_key.lower()[:1],
            args.rewind_step_key.lower()[:1],
            args.rewind_credit_key.lower()[:1],
            args.takeover_key.lower()[:1],
            args.actor_switch_key.lower()[:1],
        }
        if len(edge_keys) != 5:
            raise ValueError("rewind/takeover/actor-switch keys must be distinct")
        if edge_keys & set(KeyboardMonitor.REWARD_KEYS):
            raise ValueError("rewind/takeover/actor-switch keys must not collide with reward keys")

    tunnel: Optional[SSHTunnel] = None
    if args.ssh_host:
        if args.server_host not in ("127.0.0.1", "localhost"):
            raise ValueError("--ssh-host 时 --server-host 必须是 127.0.0.1")
        tunnel = SSHTunnel(
            host=args.ssh_host,
            local_port=args.server_port,
            remote_port=args.remote_port,
            jump=args.ssh_jump,
            connect_timeout=args.ssh_timeout,
        )
    client: Optional[DobotStage2RLTClient] = None
    policy = None
    try:
        # Same lifecycle as run_stage2_env_client: no hardware startup before
        # SSH is ready, and Ctrl-C during SSH startup still closes the child.
        if tunnel is not None:
            tunnel.ensure()
        while True:
            try:
                if tunnel is not None:
                    tunnel.ensure()
                print(f"连接 Stage 2 RLT 服务器 {args.server_host}:{args.server_port} ...")
                policy = _websocket_client_policy.WebsocketClientPolicy(
                    host=args.server_host, port=args.server_port
                )
                # WebsocketClientPolicy.reset() is a no-op. Send the RLT reset
                # explicitly so a crashed client's unfinished episode cannot
                # be joined to the new one. Weights and stored replay survive.
                reply = policy.infer({"rlt/request": "reset"})
                if not reply.get("ok", False):
                    raise RuntimeError(f"Server rejected session reset: {reply}")
                if client is None:
                    client = DobotStage2RLTClient(args, policy)
                else:
                    # Hardware is already up; only the websocket was lost.
                    client.policy = policy
                    client._connection_error = None
                print("Stage 2 RLT 服务器已连接（倒车 + 接管已启用）")
                client.run()
                return 0
            except KeyboardInterrupt:
                raise
            except NETWORK_ERRORS as exc:
                if tunnel is not None:
                    tunnel.invalidate_adopted()
                if client is not None:
                    client._stop_takeover_thread()
                    client.keyboard.stop()
                _close_policy_connection(policy)
                policy = None
                print(f"连接中断：{exc}；{args.reconnect_delay:.1f}s 后重连")
                time.sleep(args.reconnect_delay)
    except KeyboardInterrupt:
        print("\n停止 Stage 2 Dobot RLT 客户端")
        return 0
    except Exception:
        traceback.print_exc()
        return 1
    finally:
        _close_policy_connection(policy)
        if client is not None:
            client.close()
        stop_cameras()
        cv2.destroyAllWindows()
        if tunnel is not None:
            tunnel.close()


def _close_policy_connection(policy: Any) -> None:
    # The current OpenPI client exposes no public close() method.
    connection = getattr(policy, "_ws", None)
    if connection is not None:
        try:
            connection.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main(tyro.cli(Args)))
