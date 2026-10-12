#!/usr/bin/env python3
"""Standalone, low-speed leader-to-follower control for collision recovery.

This script intentionally does not use a launch node, ZMQ, cameras, or dataset
code.  It controls only the six arm joints; grippers are left unchanged.
"""

import argparse
import configparser
import signal
import socket
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

ARM_CONFIG = {
    "left": ("HAND_LEFT", "192.168.5.1"),
    "right": ("HAND_RIGHT", "192.168.5.2"),
}
UNLOCK_TAP_MAX_SECONDS = 0.5


class DashboardConnection:
    """Small subset of the Dobot dashboard protocol needed for recovery."""

    def __init__(self, ip: str):
        self._socket = socket.create_connection((ip, 29999), timeout=3.0)

    def command(self, text: str) -> str:
        self._socket.sendall(text.encode("utf-8"))
        reply = self._socket.recv(1024)
        if not reply:
            raise ConnectionError("Dobot Dashboard 连接已关闭")
        return reply.decode("utf-8", errors="replace")

    def close(self) -> None:
        self._socket.close()


class MoveConnection:
    """Small subset of the Dobot motion protocol needed for ServoJ."""

    def __init__(self, ip: str):
        self._socket = socket.create_connection((ip, 30003), timeout=3.0)

    def servo_j(self, joints_deg: np.ndarray, period: float) -> None:
        values = ",".join("{:.6f}".format(value) for value in joints_deg)
        command = "ServoJ({},{:.6f},gain=500)".format(values, period)
        self._socket.sendall(command.encode("utf-8"))

    def close(self) -> None:
        self._socket.close()


class EmergencyLeader:
    """Minimal leader reader without camera, launch, agent, or gripper code."""

    def __init__(self, section: configparser.SectionProxy):
        # Keep hardware-only dependencies lazy so ``--help`` works on any host.
        from third_party.DynamixelSDK.python.src.dynamixel_sdk.group_sync_read import (
            GroupSyncRead,
        )
        from third_party.DynamixelSDK.python.src.dynamixel_sdk.packet_handler import (
            PacketHandler,
        )
        from third_party.DynamixelSDK.python.src.dynamixel_sdk.port_handler import (
            PortHandler,
        )
        from third_party.DynamixelSDK.python.src.dynamixel_sdk.robotis_def import (
            COMM_SUCCESS,
        )

        self._comm_success = COMM_SUCCESS
        joint_ids = tuple(int(value) for value in section["joint_ids"].split(","))
        self._offsets = np.asarray(
            [float(value) for value in section["joint_offsets"].split(",")]
        )
        self._signs = np.asarray(
            [int(value) for value in section["joint_signs"].split(",")]
        )
        if len(joint_ids) != 6 or self._offsets.shape != (6,) or self._signs.shape != (6,):
            raise ValueError("主臂配置必须包含 6 个关节")
        self._joint_ids = joint_ids
        self._append_id = int(section["append_id"])
        self._port = PortHandler(section["port"])
        self._packet = PacketHandler(2.0)
        self._reader = GroupSyncRead(
            self._port, self._packet, 140, 4
        )
        if not self._port.openPort():
            raise RuntimeError("无法打开主臂串口 {}".format(section["port"]))
        if not self._port.setBaudRate(int(section["baud_rate"])):
            self._port.closePort()
            raise RuntimeError("无法设置主臂串口波特率")
        for joint_id in self._joint_ids:
            if not self._reader.addParam(joint_id):
                self.close()
                raise RuntimeError("无法读取主臂电机 ID {}".format(joint_id))

    def read(self) -> np.ndarray:
        result = self._reader.txRxPacket()
        if result != self._comm_success:
            raise RuntimeError("读取主臂失败，通信错误码 {}".format(result))
        raw = []
        for joint_id in self._joint_ids:
            if not self._reader.isAvailable(joint_id, 140, 4):
                raise RuntimeError("主臂电机 ID {} 无位置数据".format(joint_id))
            value = self._reader.getData(joint_id, 140, 4)
            raw.append(np.int32(np.uint32(value)))
        joints = np.asarray(raw, dtype=float) / 2048.0 * np.pi
        return (joints - self._offsets) * self._signs

    def buttons_pressed(self) -> Tuple[bool, bool]:
        """Return active-low ``(unlock, recording)`` button states."""
        self._port.writePort([0xAA, 0x55, 0xAA])
        self._port.setPacketTimeout(2)
        packet = []
        while len(packet) < 2 and not self._port.isPacketTimeout():
            packet.extend(self._port.readPort(2))
        valid = (
            len(packet) == 2
            and (packet[0] | packet[1]) == 0xFF
            and (packet[0] & packet[1]) == 0x00
        )
        if not valid:
            raise RuntimeError("读取主臂按键失败")
        keys = ((packet[0] >> 4) & 0x0F, packet[0] & 0x0F)
        return keys[0] == 0, keys[1] == 0

    def set_torque(self, enabled: bool) -> None:
        value = 1 if enabled else 0
        for joint_id in self._joint_ids + (self._append_id,):
            result, error = self._packet.write1ByteTxRx(
                self._port, joint_id, 64, value
            )
            if result != self._comm_success or error != 0:
                raise RuntimeError(
                    "设置主臂电机 ID {} 扭矩失败 ({}, {})".format(
                        joint_id, result, error
                    )
                )

    def close(self) -> None:
        self._port.closePort()


def _load_hand_config() -> configparser.ConfigParser:
    config = configparser.ConfigParser()
    path = Path(__file__).resolve().parent / "dobot_config" / "dobot_settings.ini"
    if not config.read(str(path)):
        raise RuntimeError("无法读取主臂配置: {}".format(path))
    return config


def _parse_angles(reply: str) -> np.ndarray:
    """Parse GetAngle() response and return six joint angles in radians."""
    try:
        values = reply.split("{", 1)[1].split("}", 1)[0].split(",")
        joints = np.deg2rad(np.asarray(values, dtype=float))
    except (IndexError, TypeError, ValueError) as exc:
        raise RuntimeError("无法解析从臂关节角度: {!r}".format(reply)) from exc
    if joints.shape != (6,) or not np.all(np.isfinite(joints)):
        raise RuntimeError("从臂返回了无效关节角度: {!r}".format(reply))
    return joints


def _nearest_equivalent(target: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Choose the target's closest 2*pi-equivalent representation."""
    return target - 2.0 * np.pi * np.round((target - reference) / (2.0 * np.pi))


def emergency_master_follow(
    arms: Iterable[str] = ("left", "right"),
    frequency: float = 25.0,
    max_step_deg: float = 0.5,
    clear_error: bool = True,
) -> None:
    """Directly and slowly command follower arms from their matching leaders.

    Args:
        arms: Any subset of ``("left", "right")``.
        frequency: Command frequency in Hz.
        max_step_deg: Maximum change sent to any joint in one command cycle.
        clear_error: Clear follower alarms before enabling motion.

    Tap a leader's unlock button to toggle its torque lock. Hold its active-low
    recording button to control the follower; releasing the recording button
    stops follower commands and restores the manually selected lock state.
    Stop the function with Ctrl+C.
    """
    selected = tuple(arms)
    unknown = set(selected) - set(ARM_CONFIG)
    if not selected or unknown:
        raise ValueError("arms 必须是 left/right，收到: {!r}".format(selected))
    if frequency <= 0:
        raise ValueError("frequency 必须大于 0")
    if not 0 < max_step_deg <= 5:
        raise ValueError("max_step_deg 必须在 (0, 5] 度之间")

    hand_configs = _load_hand_config()
    leaders: Dict[str, EmergencyLeader] = {}
    dashboards: Dict[str, DashboardConnection] = {}
    movers: Dict[str, MoveConnection] = {}
    commanded: Dict[str, np.ndarray] = {}
    leader_origins: Dict[str, np.ndarray] = {}
    follower_origins: Dict[str, np.ndarray] = {}
    intervening = {arm: False for arm in selected}
    manually_unlocked = {arm: False for arm in selected}
    unlock_button_down = {arm: False for arm in selected}
    unlock_pressed_at = {arm: 0.0 for arm in selected}
    period = 1.0 / frequency
    max_step = np.deg2rad(max_step_deg)
    stop_requested = False

    def request_stop(_signum=None, _frame=None):
        nonlocal stop_requested
        stop_requested = True

    previous_sigterm = signal.signal(signal.SIGTERM, request_stop)
    try:
        for arm in selected:
            hand_name, robot_ip = ARM_CONFIG[arm]
            print("[{}] 连接主臂 {} ...".format(arm, hand_configs[hand_name]["port"]))
            leader = EmergencyLeader(hand_configs[hand_name])
            leaders[arm] = leader
            leader.set_torque(True)

            print("[{}] 直连从臂 {} ...".format(arm, robot_ip))
            dashboard = DashboardConnection(robot_ip)
            dashboards[arm] = dashboard
            mover = MoveConnection(robot_ip)
            movers[arm] = mover
            dashboard.command("StopDrag()")
            if clear_error:
                print(
                    "[{}] 清除碰撞/报警: {}".format(
                        arm, dashboard.command("ClearError()")
                    )
                )
                time.sleep(0.2)
            print(
                "[{}] 使能从臂: {}".format(
                    arm, dashboard.command("EnableRobot()")
                )
            )
            dashboard.command("SpeedFactor(5)")
            dashboard.command("AccJ(5)")
            dashboard.command("SpeedJ(5)")
            commanded[arm] = _parse_angles(dashboard.command("GetAngle()"))

        print(
            "应急跟随待命（{}，{:.1f} Hz，每周期最大 {:.2f}°）。"
            "短按解锁键可切换主臂锁定状态；按住录制键接管，松开停止跟随；"
            "Ctrl+C 退出。".format(
                "/".join(selected), frequency, max_step_deg
            )
        )
        while not stop_requested:
            cycle_start = time.monotonic()
            for arm in selected:
                unlock_pressed, recording_pressed = leaders[arm].buttons_pressed()

                if unlock_pressed and not unlock_button_down[arm]:
                    unlock_button_down[arm] = True
                    unlock_pressed_at[arm] = time.monotonic()
                elif not unlock_pressed and unlock_button_down[arm]:
                    unlock_button_down[arm] = False
                    press_duration = time.monotonic() - unlock_pressed_at[arm]
                    if press_duration <= UNLOCK_TAP_MAX_SECONDS:
                        next_unlocked = not manually_unlocked[arm]
                        # Recording-button intervention also requires zero
                        # torque, so a lock request takes effect when that
                        # intervention ends if both buttons overlap.
                        leaders[arm].set_torque(
                            not (next_unlocked or intervening[arm])
                        )
                        manually_unlocked[arm] = next_unlocked
                        if next_unlocked:
                            print("[{}] 解锁键点按：主臂已解锁".format(arm))
                        elif intervening[arm]:
                            print(
                                "[{}] 解锁键点按：停止跟随后主臂将锁定".format(
                                    arm
                                )
                            )
                        else:
                            print("[{}] 解锁键点按：主臂已锁定".format(arm))

                pressed = recording_pressed
                if pressed and not intervening[arm]:
                    leaders[arm].set_torque(False)
                    leader_origins[arm] = leaders[arm].read()
                    commanded[arm] = _parse_angles(
                        dashboards[arm].command("GetAngle()")
                    )
                    follower_origins[arm] = commanded[arm].copy()
                    intervening[arm] = True
                    print("[{}] 录制键按下：主臂已解锁，开始人工接管".format(arm))
                elif not pressed and intervening[arm]:
                    leaders[arm].set_torque(not manually_unlocked[arm])
                    intervening[arm] = False
                    state = "保持解锁" if manually_unlocked[arm] else "已锁定"
                    print(
                        "[{}] 录制键松开：停止跟随，主臂{}".format(arm, state)
                    )

                if not intervening[arm]:
                    continue

                leader_now = np.asarray(leaders[arm].read(), dtype=float)
                if leader_now.shape != (6,) or not np.all(np.isfinite(leader_now)):
                    raise RuntimeError("[{}] 主臂返回了无效关节角度".format(arm))
                leader_delta = _nearest_equivalent(
                    leader_now, leader_origins[arm]
                ) - leader_origins[arm]
                target = follower_origins[arm] + leader_delta
                delta = np.clip(target - commanded[arm], -max_step, max_step)
                commanded[arm] = commanded[arm] + delta
                degrees = np.rad2deg(commanded[arm])
                movers[arm].servo_j(degrees, period=max(0.02, period))

            remaining = period - (time.monotonic() - cycle_start)
            if remaining > 0:
                time.sleep(remaining)
    except KeyboardInterrupt:
        request_stop()
    finally:
        print("正在停止应急跟随并锁定主臂 ...")
        for leader in leaders.values():
            try:
                leader.set_torque(True)
            except Exception as exc:
                print("[警告] 主臂恢复扭矩失败: {}".format(exc), file=sys.stderr)
            try:
                leader.close()
            except Exception:
                pass
        for connection in tuple(movers.values()) + tuple(dashboards.values()):
            try:
                connection.close()
            except Exception:
                pass
        signal.signal(signal.SIGTERM, previous_sigterm)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="不启动 launch/ZMQ，直接用主臂低速带动从臂脱离碰撞位置。"
    )
    parser.add_argument(
        "--arm",
        choices=("left", "right", "both"),
        default="both",
        help="控制左臂、右臂或双臂（默认: both）",
    )
    parser.add_argument("--frequency", type=float, default=25.0, help="控制频率 Hz")
    parser.add_argument(
        "--max-step-deg",
        type=float,
        default=0.5,
        help="每周期每关节最大变化角度（默认: 0.5°，最大: 5°）",
    )
    parser.add_argument(
        "--no-clear-error",
        action="store_true",
        help="不发送 ClearError（通常碰撞恢复不应使用此参数）",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    arms: Tuple[str, ...] = ("left", "right") if args.arm == "both" else (args.arm,)
    print("确认人员已离开机械臂工作区，急停按钮可随时触达。3 秒后开始连接。")
    try:
        time.sleep(3)
        emergency_master_follow(
            arms=arms,
            frequency=args.frequency,
            max_step_deg=args.max_step_deg,
            clear_error=not args.no_clear_error,
        )
    except (ConnectionError, ImportError, OSError, RuntimeError, ValueError) as exc:
        print("应急跟随启动/运行失败: {}".format(exc), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
