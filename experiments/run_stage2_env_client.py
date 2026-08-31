#!/usr/bin/env python3
"""Standalone RLT Stage 2 Dobot client with continuous per-arm intervention.

Human actions executed while Stage 2 is inferring are queued.  The next RPC
returns those real actions before accepting a policy inferred from stale state.
The wire format remains fixed-size so the existing Stage 2 replay schema works.

This file is self-contained: the RPC wire protocol, keyboard reward feedback,
RealSense capture threads and the Dobot environment client are all inlined, so
it drives the robot without importing any other experiments module.
"""
from __future__ import annotations

import os
import select
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import termios
import threading
import time
import traceback
import tty
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import numpy as np
import tyro
from openpi_client import msgpack_numpy

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)
from dobot_control.agents.agent import BimanualAgent
from dobot_control.agents.dobot_agent import DobotAgent
from dobot_control.cameras.realsense_camera import RealSenseCamera
from dobot_control.env import RobotEnv
from dobot_control.robots.robot_node import ZMQClientRobot
from scripts.manipulate_utils import load_ini_data_camera, load_ini_data_hands


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


# --- Stage 2 RPC wire protocol ---

_HEADER = struct.Struct("!Q")


def recv_exact(connection: socket.socket, size: int) -> bytes:
    chunks = []
    while size:
        chunk = connection.recv(size)
        if not chunk:
            raise ConnectionError("Stage 2 server disconnected")
        chunks.append(chunk)
        size -= len(chunk)
    return b"".join(chunks)


def recv_message(connection: socket.socket) -> dict[str, Any]:
    size = _HEADER.unpack(recv_exact(connection, _HEADER.size))[0]
    if size > 512 * 1024 * 1024:
        raise RuntimeError(f"Refusing oversized RPC message: {size} bytes")
    return msgpack_numpy.unpackb(recv_exact(connection, size))


def send_message(connection: socket.socket, message: dict[str, Any]) -> None:
    payload = msgpack_numpy.Packer().pack(message)
    connection.sendall(_HEADER.pack(len(payload)) + payload)


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

        Connecting would open a real RPC connection through the tunnel that the
        Stage 2 server would see and immediately lose.
        """
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
        """Start the tunnel, or rebuild it if the ssh child has died."""
        if self._adopted:
            return
        if self._process is not None and self._process.poll() is None:
            return
        if self._process is not None:
            print(f"SSH 隧道已断开（ssh 退出码 {self._process.returncode}），正在重建 ...")
        elif self._port_taken():
            # A tunnel from another terminal already forwards this port; reuse
            # it instead of fighting over the bind.
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


class KeyboardReward:
    def __init__(self) -> None:
        self._old_settings = None
        self._raw = False
        self._terminal_signal: Optional[str] = None

    def start(self) -> None:
        self._terminal_signal = None
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
        if self._terminal_signal is not None:
            return self._terminal_signal
        if self._raw and select.select([sys.stdin], [], [], 0)[0]:
            key = sys.stdin.read(1).lower()
            if key == "\x03":
                raise KeyboardInterrupt
            if key in ("s", " "):
                self._terminal_signal = "s"
            elif key == "f":
                self._terminal_signal = "f"
            elif key == "r":
                self._terminal_signal = "r"
            elif key == "a":
                return "a"
            elif key == "p":
                return "p"
            elif key == "b":
                return "b"
        return self._terminal_signal


@dataclass
class Args:
    server_host: str = "127.0.0.1"
    server_port: int = 18000
    # Set --ssh-host to let this process own the ssh -N -L tunnel itself.
    # Left unset, nothing changes: connect to server_host:server_port as before.
    ssh_host: Optional[str] = None
    ssh_jump: Optional[str] = None
    # Port the Stage 2 server listens on, on the far side of the tunnel.
    remote_port: int = 8000
    ssh_timeout: float = 30.0
    robot_port: int = 6001
    hostname: str = "127.0.0.1"
    instruction: str = "pour water"
    control_hz: float = 10.0
    always_actor: bool = False
    crop_top_camera: bool = False
    dry_run: bool = False
    reconnect_delay: float = 3.0
    max_pending_control_steps: int = 100


@dataclass
class ExecutedStep:
    action: np.ndarray
    intervention: np.ndarray
    reward: float
    done: bool
    success: Optional[bool]
    observation: dict[str, Any]
    actor_switch_requested: bool
    discard_episode_requested: bool
    anchor_requested: bool


@dataclass
class ControlJob:
    actions: np.ndarray
    hold_only: bool = False
    records: list[ExecutedStep] = field(default_factory=list)
    finished: bool = False
    error: Optional[BaseException] = None


def make_leader_agent() -> BimanualAgent:
    _, hands = load_ini_data_hands()
    return BimanualAgent(
        DobotAgent(which_hand="LEFT", dobot_config=hands["HAND_LEFT"]),
        DobotAgent(which_hand="RIGHT", dobot_config=hands["HAND_RIGHT"]),
    )


class DobotStage2InterventionClient:
    def __init__(self, args: Args) -> None:
        self.args = args
        self.feedback = KeyboardReward()
        self.prompt = args.instruction
        self.last_action: Optional[np.ndarray] = None
        self.last_observation: Optional[dict[str, Any]] = None
        self.robot_faulted = False
        init_cameras(crop_top_camera=args.crop_top_camera)
        robot_client = ZMQClientRobot(port=args.robot_port, host=args.hostname)
        self.env = RobotEnv(robot_client)
        if not args.dry_run:
            for channel in (1, 2, 3):
                self.env.set_do_status([channel, 0])
        if args.max_pending_control_steps <= 0:
            raise ValueError("max_pending_control_steps must be positive")
        self.leader = make_leader_agent()
        self.leader.set_torque(2, True)
        self.was_intervening = np.array([False, False])
        self.leader_unlocked = np.array([False, False])
        self._button_a_pressed = np.array([False, False])
        self._button_a_pressed_at = np.zeros(2, dtype=np.float64)
        self.leader_origin = np.zeros(14, dtype=np.float64)
        self.follower_origin = np.zeros(14, dtype=np.float64)

        self._condition = threading.Condition()
        self._env_lock = threading.RLock()
        self._leader_state_lock = threading.RLock()
        self._pending: deque[ExecutedStep] = deque()
        self._job: Optional[ControlJob] = None
        self._background_error: Optional[BaseException] = None
        self._tick_in_progress = False
        self._episode_done = False
        self._resetting = True
        # Rewind anchor: a mid-episode state the operator can restart from.
        self._anchor_lock = threading.Lock()
        self._anchor: Optional[dict[str, Any]] = None
        self._step_index = 0
        self._stop_event = threading.Event()
        self._button_thread = threading.Thread(
            target=self._button_loop, name="dobot-button-a-monitor", daemon=True
        )
        self._control_thread = threading.Thread(
            target=self._control_loop, name="dobot-stage2-control", daemon=True
        )
        self._button_thread.start()
        self._control_thread.start()

    def close(self) -> None:
        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()
        self._button_thread.join(timeout=5.0)
        self._control_thread.join(timeout=5.0)
        self.feedback.stop()
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

    def _read_observation(self) -> dict[str, Any]:
        images = get_current_images()
        state = np.asarray(self.env.get_obs()["joint_positions"], dtype=np.float32).copy()
        if self.last_action is not None:
            state[6] = self.last_action[6]
            state[13] = self.last_action[13]
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

    def _reset_env(self, prompt: str) -> dict[str, Any]:
        self.feedback.stop()
        self.prompt = prompt or self.args.instruction
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
        self.last_action[6] = 1.0
        self.last_action[13] = 1.0
        observation = self.observation()
        self.feedback.start()
        self._print_episode_banner()
        return observation

    def reset(self, prompt: str) -> dict[str, Any]:
        with self._condition:
            self._resetting = True
            while self._job is not None:
                self._condition.wait()
        try:
            with self._leader_state_lock:
                self.leader.set_torque(2, True)
                self.was_intervening[:] = False
                self.leader_unlocked[:] = False
                self._button_a_pressed[:] = False
                self._button_a_pressed_at[:] = 0.0
            self.leader_origin[:] = 0.0
            self.follower_origin[:] = 0.0
            self.feedback.stop()
            anchor = self._offer_rewind()
            with self._env_lock:
                if anchor is None:
                    observation = self._reset_env(prompt)
                else:
                    observation = self._rewind_to_anchor(anchor, prompt)
            with self._condition:
                self._pending.clear()
                self._background_error = None
                self._episode_done = False
                self._step_index = 0
            print(
                "人工介入：按住任一主手的录制键接管，松开后等待新 policy；"
                "短按主手上的 A 键可解锁/锁定对应主臂以恢复初始位置"
                "（键盘 a 是标记倒车 anchor，两者无关）"
            )
            return observation
        finally:
            with self._condition:
                self._resetting = False
                self._condition.notify_all()

    def _print_episode_banner(self) -> None:
        keys = (
            "s/空格=成功，f=失败，r=丢弃本回合，p=进展(+0.5)，"
            "a=标记倒车 anchor，Ctrl+C=停止"
        )
        if self.args.always_actor:
            print(f"回合开始（全程 actor）：{keys}")
        else:
            print(f"回合开始（base VLA）：b=下一 chunk 切换 actor，{keys}")

    def _mark_anchor(self, action: np.ndarray) -> None:
        """Snapshot the current follower command as a rewind point."""
        snapshot = {
            "action": np.asarray(action, dtype=np.float64).copy(),
            "step_index": self._step_index,
            "created_time": time.time(),
        }
        with self._anchor_lock:
            self._anchor = snapshot
        print(
            f"已标记倒车 anchor（本回合第 {self._step_index} 步）；"
            "回合结束后可选择回到该状态重开"
        )

    def _clear_anchor(self) -> None:
        with self._anchor_lock:
            self._anchor = None

    def _offer_rewind(self) -> Optional[dict[str, Any]]:
        """Ask whether the next episode should restart from the rewind anchor."""
        with self._anchor_lock:
            anchor = self._anchor
        if anchor is None:
            return None
        if self.robot_faulted:
            self._clear_anchor()
            print("机械臂故障，倒车 anchor 已作废；本回合走正常复位")
            return None
        age = time.time() - float(anchor["created_time"])
        choice = input(
            f"检测到倒车 anchor（第 {int(anchor['step_index'])} 步，{age:.0f}s 前）。"
            "直接 Enter=倒车到该 anchor 开新回合，输入 n=正常复位："
        ).strip().lower()
        if choice.startswith("n"):
            self._clear_anchor()
            print("已放弃倒车 anchor，执行正常复位")
            return None
        return anchor

    def _rewind_to_anchor(
        self, anchor: dict[str, Any], prompt: str
    ) -> dict[str, Any]:
        """Drive the follower back to the anchor and start a fresh episode there.

        The anchor is kept so the same state can be replayed repeatedly.  The
        episode that follows is an ordinary one: the server drives it, s/f/r and
        intervention all behave normally and it is written to the replay buffer.
        """
        target = np.asarray(anchor["action"], dtype=np.float64).reshape(-1)
        if target.shape != (14,) or not np.isfinite(target).all():
            raise ValueError("倒车 anchor 必须是有限的 14 维动作")
        self.prompt = prompt or self.args.instruction

        print("\n=== 倒车到 anchor ===")
        input("请确认夹爪无危险物体且倒车路径已清空，按 Enter 让机械臂回到 anchor：")
        try:
            if self.args.dry_run:
                self.last_action = target.copy()
            else:
                current = np.asarray(
                    self.env.get_obs()["joint_positions"], dtype=np.float64
                ).reshape(-1)
                move_linearly(self.env, current, target, max_steps=150)
                self.last_action = target.copy()
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            self.robot_faulted = True
            self._clear_anchor()
            print(f"倒车失败，改走正常复位：{exc}")
            return self._reset_env(prompt)

        input("请把物体恢复成 anchor 时的状态，确认安全后按 Enter 开始新回合：")
        observation = self._read_observation()
        self.feedback.start()
        self._print_episode_banner()
        print(
            f"本回合从倒车 anchor（原第 {int(anchor['step_index'])} 步）开始；"
            "anchor 保留，可重复倒车"
        )
        return observation

    def prepare_action(self, proposed: np.ndarray) -> np.ndarray:
        action = np.asarray(proposed, dtype=np.float64).copy()
        if action.shape != (14,) or not np.isfinite(action).all():
            raise ValueError(f"Invalid action: shape={action.shape}, finite={np.isfinite(action).all()}")
        action[6] = np.clip(action[6], 0.0, 1.0)
        action[13] = np.clip(action[13], 0.0, 1.0)
        return action

    def fault_response(
        self,
        error: BaseException,
        executed: np.ndarray,
        rewards: np.ndarray,
        steps_executed: int,
    ) -> dict[str, Any]:
        self.feedback.stop()
        self.robot_faulted = True
        message = f"{type(error).__name__}: {error}"
        print(f"机械臂故障，当前回合终止且该 transition 将被丢弃：{message}")
        return {
            "ok": True,
            "observation": self.safe_observation(),
            "executed_actions": executed,
            "rewards": rewards,
            "done": True,
            "info": {
                "success": False,
                "robot_fault": True,
                "discard_transition": True,
                "fault_message": message,
                "steps_executed": steps_executed,
            },
        }

    def _update_button_a(
        self, keys: np.ndarray, intervening: np.ndarray
    ) -> None:
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
                        # leader, so locking A must not fight an active intervention.
                        should_unlock = (
                            self.leader_unlocked[side] or intervening[side]
                        )
                        self.leader.set_torque(side, not should_unlock)
                        name = "左臂" if side == 0 else "右臂"
                        if self.leader_unlocked[side]:
                            state = "解锁，可恢复初始位置"
                        elif intervening[side]:
                            state = "恢复模式关闭；人工接管结束后锁定"
                        else:
                            state = "锁定"
                        print(
                            f"{name} A 键恢复模式：{state}"
                            f"（短按 {duration:.2f}s）"
                        )
                    self._button_a_pressed_at[side] = 0.0
            self._button_a_pressed = pressed

    def _button_loop(self) -> None:
        """Monitor button A for the entire lifetime of the client process."""
        interval = 1.0 / max(self.args.control_hz, 20.0)
        while not self._stop_event.is_set():
            keys = self.leader.get_keys()
            intervening = np.asarray(keys[:, 1] == 0, dtype=bool)
            self._update_button_a(keys, intervening)
            self._stop_event.wait(interval)

    def _intervention_action(
        self, proposed: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Map leader displacement without an artificial manual step limit."""
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
                    self.leader.set_torque(
                        side, not self.leader_unlocked[side]
                    )
                print(f"{name}归还 policy；旧 policy 已失效")

        target = proposed.copy()
        for side in range(2):
            arm = slice(side * 7, side * 7 + 7)
            if intervening[side]:
                target[arm] = (
                    self.follower_origin[arm]
                    + leader_now[arm]
                    - self.leader_origin[arm]
                )
            elif released[side]:
                target[arm] = self.last_action[arm]
        # prepare_action retains finite-value validation and legal gripper range.
        return self.prepare_action(target), intervening, released

    def _execute_tick(self, proposed: np.ndarray) -> tuple[ExecutedStep, np.ndarray]:
        action, intervening, released = self._intervention_action(proposed)
        with self._env_lock:
            if not self.args.dry_run:
                self.env.step(action, np.array([1, 1]))
            self.last_action = action.copy()
            observation = self._read_observation()
        self.was_intervening = intervening.copy()

        reward, done, success = 0.0, False, None
        actor_switch_requested = self.args.always_actor
        discard_episode_requested = False
        anchor_requested = False
        self._step_index += 1
        signal = self.feedback.check()
        if signal == "s":
            reward, done, success = 1.0, True, True
        elif signal == "f":
            done, success = True, False
        elif signal == "p":
            reward = 0.5
        elif signal == "b":
            actor_switch_requested = True
            print("已发送 actor 切换请求；若 actor 已就绪，将从下一 chunk 生效")
        elif signal == "a":
            anchor_requested = True
            self._mark_anchor(action)
        elif signal == "r":
            discard_episode_requested = True
            done = True
            print("已标记丢弃本回合；该 episode 不会写入 warmup replay")
        if done:
            self.feedback.stop()
        return ExecutedStep(
            action.copy(),
            intervening.copy(),
            reward,
            done,
            success,
            observation,
            actor_switch_requested,
            discard_episode_requested,
            anchor_requested,
        ), released

    def _control_loop(self) -> None:
        interval = 1.0 / self.args.control_hz
        deadline = time.monotonic()
        while not self._stop_event.is_set():
            with self._condition:
                if self._resetting or self._episode_done or self.last_action is None:
                    self._condition.wait(timeout=interval)
                    deadline = time.monotonic()
                    continue
                job = self._job
                self._tick_in_progress = True
            try:
                if job is None:
                    # Poll intervention edges continuously.  No follower command
                    # is sent between RPCs unless a human is actively controlling.
                    _, intervening, _ = self._intervention_action(self.last_action)
                    self.was_intervening = intervening.copy()
                    if not intervening.any():
                        with self._condition:
                            self._tick_in_progress = False
                            self._condition.notify_all()
                        deadline += interval
                        self._stop_event.wait(max(0.0, deadline - time.monotonic()))
                        continue
                    proposed = self.last_action.copy()
                else:
                    proposed = (
                        self.last_action.copy()
                        if job.hold_only
                        else job.actions[len(job.records)]
                    )

                record, released = self._execute_tick(proposed)
                with self._condition:
                    if record.done:
                        self._episode_done = True
                    if job is None:
                        if len(self._pending) >= self.args.max_pending_control_steps:
                            raise RuntimeError("pending intervention queue is full")
                        self._pending.append(record)
                    else:
                        job.records.append(record)
                        if released.any():
                            job.hold_only = True
                        if record.done or len(job.records) == len(job.actions):
                            job.finished = True
                            self._job = None
                    self._tick_in_progress = False
                    self._condition.notify_all()
            except BaseException as exc:
                with self._condition:
                    self._episode_done = True
                    if job is None:
                        self._background_error = exc
                    else:
                        job.error, job.finished = exc, True
                        self._job = None
                    self._tick_in_progress = False
                    self._condition.notify_all()
                with self._leader_state_lock:
                    self.leader.set_torque(2, True)
                    self.was_intervening[:] = False
                    self.leader_unlocked[:] = False

            deadline += interval
            self._stop_event.wait(max(0.0, deadline - time.monotonic()))
            if time.monotonic() - deadline > interval:
                deadline = time.monotonic()

    def _collect_transition(
        self, actions: np.ndarray
    ) -> tuple[list[ExecutedStep], bool]:
        """Atomically drain queued manual steps or publish a policy job."""
        with self._condition:
            while self._tick_in_progress:
                self._condition.wait()
            if self._background_error is not None:
                error, self._background_error = self._background_error, None
                raise error
            if self._job is not None:
                raise RuntimeError("control job already active")

            records: list[ExecutedStep] = []
            while self._pending and len(records) < len(actions):
                record = self._pending.popleft()
                records.append(record)
                if record.done:
                    break
            stale = bool(records)
            if records and (records[-1].done or len(records) == len(actions)):
                return records, stale

            if records:
                # The proposed policy was inferred from the observation before
                # these queued human actions. Complete with hold commands.
                remaining = len(actions) - len(records)
                job_actions = np.repeat(self.last_action[None], remaining, axis=0)
                job = ControlJob(actions=job_actions, hold_only=True)
            else:
                job = ControlJob(actions=actions, hold_only=False)
            self._job = job
            self._condition.notify_all()
            while not job.finished and not self._stop_event.is_set():
                self._condition.wait()
        if job.error is not None:
            raise job.error
        if not job.finished:
            raise RuntimeError("control thread stopped")
        records.extend(job.records)
        return records, stale

    def _make_response(
        self, records: list[ExecutedStep], chunk_length: int, stale: bool
    ) -> dict[str, Any]:
        if not records:
            raise RuntimeError("no action was executed")
        executed = np.repeat(records[-1].action[None], chunk_length, axis=0)
        rewards = np.zeros(chunk_length, dtype=np.float32)
        mask = np.zeros((chunk_length, 2), dtype=bool)
        for index, record in enumerate(records):
            executed[index], rewards[index], mask[index] = (
                record.action, record.reward, record.intervention
            )
        last = records[-1]
        info: dict[str, Any] = {
            "steps_executed": len(records),
            "intervention_mask": mask,
            "intervention_occurred": bool(mask[:len(records)].any()),
            "action_chunk_truncated": False,
            "stale_policy_discarded": stale,
            "actor_switch_requested": any(
                record.actor_switch_requested for record in records
            ),
            "discard_episode": any(
                record.discard_episode_requested for record in records
            ),
            "anchor_marked": any(record.anchor_requested for record in records),
        }
        if last.success is not None:
            info["success"] = last.success
        print(
            f"chunk steps={len(records)}, reward={rewards.sum():.1f}, "
            f"done={last.done}, intervention={info['intervention_occurred']}, "
            f"stale_policy_discarded={stale}, "
            f"actor_switch_requested={info['actor_switch_requested']}"
        )
        return {"ok": True, "observation": last.observation,
                "executed_actions": executed, "rewards": rewards,
                "done": last.done, "info": info}

    def step(self, proposed_actions: np.ndarray) -> dict[str, Any]:
        actions = np.asarray(proposed_actions, dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] != 14:
            raise ValueError(f"Expected actions [C,14], got {actions.shape}")
        if self.last_action is None:
            raise RuntimeError("reset must complete before step")
        try:
            records, stale = self._collect_transition(actions)
            return self._make_response(records, len(actions), stale)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            executed = np.repeat(self.last_action[None], len(actions), axis=0)
            return self.fault_response(
                exc, executed, np.zeros(len(actions), dtype=np.float32), 0
            )

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        command = request.get("command")
        if command == "reset":
            return {"ok": True, "observation": self.reset(str(request.get("prompt", "")))}
        if command == "step":
            return self.step(request["actions"])
        raise ValueError(f"Unknown command: {command!r}")


def main(args: Args) -> int:
    if args.control_hz <= 0:
        raise ValueError("control_hz must be positive")
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
        # Fail before the cameras and the robot are brought up.
        tunnel.ensure()
    client: Optional[DobotStage2InterventionClient] = None
    try:
        client = DobotStage2InterventionClient(args)
        while True:
            try:
                if tunnel is not None:
                    tunnel.ensure()
                print(f"连接 Stage 2 环境服务器 {args.server_host}:{args.server_port} ...")
                with socket.create_connection(
                    (args.server_host, args.server_port), timeout=30
                ) as connection:
                    connection.settimeout(None)
                    print("Stage 2 环境服务器已连接（连续人工介入已启用）")
                    while True:
                        request = recv_message(connection)
                        try:
                            response = client.handle(request)
                        except KeyboardInterrupt:
                            raise
                        except Exception:
                            response = {"ok": False, "error": traceback.format_exc()}
                        send_message(connection, response)
            except KeyboardInterrupt:
                raise
            except (ConnectionError, OSError) as exc:
                print(f"连接中断：{exc}；{args.reconnect_delay:.1f}s 后重连")
                time.sleep(args.reconnect_delay)
    except KeyboardInterrupt:
        print("\n停止 Stage 2 人工介入真机客户端")
        return 0
    finally:
        if client is not None:
            client.close()
        stop_cameras()
        cv2.destroyAllWindows()
        if tunnel is not None:
            tunnel.close()


if __name__ == "__main__":
    raise SystemExit(main(tyro.cli(Args)))
