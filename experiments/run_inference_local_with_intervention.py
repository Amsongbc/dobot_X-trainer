#!/usr/bin/env python3
"""Original local-checkpoint inference with per-arm human intervention.

This is the intervention counterpart of ``run_inference.py``. Policy inference
runs locally through ``Imitate_Model``. Hold a leader hand's recording button
to take over its follower arm; release it to return that arm to the local
policy.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from dataclasses import dataclass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "ModelTrain"))

import numpy as np
import tyro

from dobot_control.agents.agent import BimanualAgent
from dobot_control.agents.dobot_agent import DobotAgent
from dobot_control.cameras.realsense_camera import RealSenseCamera
from dobot_control.env import RobotEnv
from dobot_control.robots.robot_node import ZMQClientRobot
from scripts.manipulate_utils import load_ini_data_camera, load_ini_data_hands


@dataclass
class Args:
    robot_port: int = 6001
    hostname: str = "127.0.0.1"
    show_img: bool = False
    # True: use the virtual jitter model; False: load the real checkpoint.
    test_control: bool = True
    virtual_jitter: float = 0.002
    control_hz: float = 10.0
    ckpt_dir: str = "./ckpt/clean_dishes"
    ckpt_name: str = "clean_dishes_model.ckpt"


images = [None, None, None]  # top, left wrist, right wrist
image_lock = threading.Lock()
camera_running = False


class VirtualJitterModel:
    """A cheap policy used to isolate inference latency from teleoperation."""

    def __init__(self, amplitude: float) -> None:
        self.amplitude = amplitude

    def predict(self, observation: dict, step: int) -> np.ndarray:
        action = np.asarray(observation["qpos"], dtype=np.float64).copy()
        # Alternate around the measured pose instead of a fixed startup pose, so
        # releasing intervention cannot make the follower jump back unexpectedly.
        direction = 1.0 if step % 2 == 0 else -1.0
        action[[5, 12]] += direction * self.amplitude
        return action


def camera_loop(camera: RealSenseCamera, index: int) -> None:
    global camera_running
    while camera_running:
        image, _ = camera.read()
        image = image[:, :, ::-1]
        with image_lock:
            images[index] = image


def start_cameras() -> None:
    global camera_running
    config = load_ini_data_camera()
    cameras = (
        RealSenseCamera(flip=True, device_id=config["top"]),
        RealSenseCamera(flip=False, device_id=config["left"]),
        RealSenseCamera(flip=True, device_id=config["right"]),
    )
    camera_running = True
    for index, camera in enumerate(cameras):
        threading.Thread(
            target=camera_loop,
            args=(camera, index),
            daemon=True,
        ).start()
    deadline = time.time() + 10.0
    while True:
        with image_lock:
            ready = all(image is not None for image in images)
        if ready:
            return
        if time.time() >= deadline:
            raise RuntimeError("Timed out waiting for camera images")
        time.sleep(0.02)


def current_images() -> list[np.ndarray]:
    with image_lock:
        if any(image is None for image in images):
            raise RuntimeError("Camera frames are not ready")
        return [image.copy() for image in images]


def make_leader_agent() -> BimanualAgent:
    _, hands = load_ini_data_hands()
    return BimanualAgent(
        DobotAgent(which_hand="LEFT", dobot_config=hands["HAND_LEFT"]),
        DobotAgent(which_hand="RIGHT", dobot_config=hands["HAND_RIGHT"]),
    )


def move_linearly(env: RobotEnv, target: np.ndarray, max_step: float = 0.001) -> None:
    current = env.get_obs()["joint_positions"]
    steps = max(1, min(int(np.max(np.abs(target - current)) / max_step) + 1, 150))
    for command in np.linspace(current, target, steps):
        env.step(command, np.array([1, 1]))


def main(args: Args) -> int:
    global camera_running
    leader = None
    cv2 = None
    try:
        if args.virtual_jitter < 0:
            raise ValueError("virtual_jitter must be non-negative")
        if args.control_hz <= 0:
            raise ValueError("control_hz must be positive")
        if args.show_img:
            import cv2 as cv2_module

            cv2 = cv2_module

        use_cameras = not args.test_control or args.show_img
        if use_cameras:
            start_cameras()
        env = RobotEnv(ZMQClientRobot(port=args.robot_port, host=args.hostname))
        for channel in (1, 2, 3):
            env.set_do_status([channel, 0])

        safe_left = np.deg2rad([-90, 30, -110, 20, 90, 90, 0])
        safe_right = np.deg2rad([90, -30, 110, -20, -90, -90, 0])
        move_linearly(env, np.concatenate([safe_left, safe_right]))
        time.sleep(1)
        photo_left = np.deg2rad([-90, 0, -90, 0, 90, 90, 57])
        photo_right = np.deg2rad([90, 0, 90, 0, -90, -90, 57])
        move_linearly(env, np.concatenate([photo_left, photo_right]))

        if args.test_control:
            model = VirtualJitterModel(args.virtual_jitter)
            print(
                f"虚拟模型已启用：腕关节抖动幅度={args.virtual_jitter:.4f} rad，"
                f"控制频率={args.control_hz:.1f} Hz"
            )
        else:
            from ModelTrain.module.model_module import Imitate_Model

            model = Imitate_Model(ckpt_dir=args.ckpt_dir, ckpt_name=args.ckpt_name)
            model.loadModel()
            print("真实本地模型已加载")
        leader = make_leader_agent()
        leader.set_torque(2, True)

        obs = env.get_obs()
        obs["joint_positions"][[6, 13]] = 1.0
        observation = {
            "qpos": obs["joint_positions"].copy(),
            "images": {"left_wrist": None, "right_wrist": None, "top": None},
        }
        last_command = observation["qpos"].copy()
        was_intervening = np.array([False, False])
        leader_origin = np.zeros(14)
        follower_origin = np.zeros(14)

        print("本地 policy 已启动：按住对应主手录制键进行人工接管，松开恢复 policy")
        t = 0
        interval = 1.0 / args.control_hz
        while True:
            started = time.monotonic()
            if use_cameras:
                frames = current_images()
                observation["images"]["top"] = frames[0]
                observation["images"]["left_wrist"] = frames[1]
                observation["images"]["right_wrist"] = frames[2]
                if cv2 is not None:
                    cv2.imshow("local policy / intervention", np.hstack(frames))
                    cv2.waitKey(1)

            policy_action = np.asarray(model.predict(observation, t), dtype=np.float64).copy()
            if policy_action.shape != (14,) or not np.isfinite(policy_action).all():
                raise RuntimeError(f"Invalid local policy action: {policy_action.shape}")
            policy_action[[6, 13]] = np.clip(policy_action[[6, 13]], 0.0, 1.0)

            keys = leader.get_keys()
            intervening = np.asarray(keys[:, 1] == 0, dtype=bool)
            released = np.logical_and(was_intervening, np.logical_not(intervening))
            leader_now = leader.act({})
            actual_now = env.get_obs()["joint_positions"]
            actual_now[[6, 13]] = last_command[[6, 13]]

            for side in range(2):
                side_slice = slice(side * 7, side * 7 + 7)
                side_name = "左臂" if side == 0 else "右臂"
                if intervening[side] and not was_intervening[side]:
                    leader.set_torque(side, False)
                    leader_now = leader.act({})
                    leader_origin[side_slice] = leader_now[side_slice]
                    follower_origin[side_slice] = actual_now[side_slice]
                    print(f"{side_name}人工接管")
                elif released[side]:
                    leader.set_torque(side, True)
                    print(f"{side_name}恢复本地 policy")

            target = policy_action.copy()
            for side in range(2):
                side_slice = slice(side * 7, side * 7 + 7)
                if intervening[side]:
                    target[side_slice] = (
                        follower_origin[side_slice]
                        + leader_now[side_slice]
                        - leader_origin[side_slice]
                    )
                elif released[side]:
                    target[side_slice] = actual_now[side_slice]

            target[[6, 13]] = np.clip(target[[6, 13]], 0.0, 1.0)
            command = target

            obs = env.step(command, np.array([1, 1]))
            obs["joint_positions"][[6, 13]] = command[[6, 13]]
            observation["qpos"] = obs["joint_positions"].copy()
            last_command = command.copy()
            was_intervening = intervening.copy()
            t += 1

            remaining = interval - (time.monotonic() - started)
            if remaining > 0:
                time.sleep(remaining)

    except KeyboardInterrupt:
        print("\n停止本地 policy 人工介入程序")
        return 0
    finally:
        camera_running = False
        if leader is not None:
            leader.set_torque(2, True)
        if cv2 is not None:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main(tyro.cli(Args)))
