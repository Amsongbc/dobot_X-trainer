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
sys.pat版本h.append(os.path.join(BASE_DIR, "ModelTrain"))

import cv2
import numpy as np
import tyro

from dobot_control.agents.agent import BimanualAgent
from dobot_control.agents.dobot_agent import DobotAgent
from dobot_control.cameras.realsense_camera import RealSenseCamera
from dobot_control.env import RobotEnv
from dobot_control.robots.robot_node import ZMQClientRobot
from ModelTrain.module.model_module import Imitate_Model
from scripts.manipulate_utils import load_ini_data_camera, load_ini_data_hands


@dataclass
class Args:
    robot_port: int = 6001
    hostname: str = "127.0.0.1"
    show_img: bool = True
    ckpt_dir: str = "./ckpt/clean_dishes"
    ckpt_name: str = "clean_dishes_model.ckpt"


images = [None, None, None]  # top, left wrist, right wrist
image_lock = threading.Lock()
camera_running = False


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


def command_is_safe(action: np.ndarray) -> bool:
    left_ok = -2.6 < action[2] < 0.0 and action[3] > -0.6
    right_ok = 0.0 < action[9] < 2.6 and action[10] < 0.6
    return left_ok and right_ok


def workspace_is_safe(env: RobotEnv) -> bool:
    position = env.get_XYZrxryrz_state()
    left_ok = (
        -410 < position[0] < 300
        and -700 < position[1] < -210
        and position[2] > 42
    )
    right_ok = (
        -250 < position[6] < 410
        and -700 < position[7] < -210
        and position[8] > 42
    )
    return left_ok and right_ok


def set_error_light(env: RobotEnv) -> None:
    env.set_do_status([3, 0])
    env.set_do_status([2, 0])
    env.set_do_status([1, 1])


def main(args: Args) -> int:
    global camera_running
    leader = None
    try:
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

        model = Imitate_Model(ckpt_dir=args.ckpt_dir, ckpt_name=args.ckpt_name)
        model.loadModel()
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
        while True:
            frames = current_images()
            observation["images"]["top"] = frames[0]
            observation["images"]["left_wrist"] = frames[1]
            observation["images"]["right_wrist"] = frames[2]
            if args.show_img:
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
            if not command_is_safe(command) or not workspace_is_safe(env):
                set_error_light(env)
                raise RuntimeError("Safety boundary reached; robot command stopped")

            obs = env.step(command, np.array([1, 1]))
            obs["joint_positions"][[6, 13]] = command[[6, 13]]
            observation["qpos"] = obs["joint_positions"].copy()
            last_command = command.copy()
            was_intervening = intervening.copy()
            t += 1

    except KeyboardInterrupt:
        print("\n停止本地 policy 人工介入程序")
        return 0
    finally:
        camera_running = False
        if leader is not None:
            leader.set_torque(2, True)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main(tyro.cli(Args)))
