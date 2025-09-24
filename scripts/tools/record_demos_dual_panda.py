# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to record demonstrations with Isaac Lab environments using human teleoperation.

This script allows users to record demonstrations operated by human teleoperation for a specified task.
The recorded demonstrations are stored as episodes in a hdf5 file. Users can specify the task, teleoperation
device, dataset directory, and environment stepping rate through command-line arguments.

required arguments:
    --task                    Name of the task.

optional arguments:
    -h, --help                Show this help message and exit
    --teleop_device           Device for interacting with environment. (default: keyboard)
    --dataset_file            File path to export recorded demos. (default: "./datasets/dataset.hdf5")
    --step_hz                 Environment stepping rate in Hz. (default: 30)
    --num_demos               Number of demonstrations to record. (default: 0)
    --num_success_steps       Number of continuous steps with task success for concluding a demo as successful. (default: 10)
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import time
import numpy as np
from quest.webrtc_headset import WebRTCHeadset
from quest.headset_control import HeadsetOurControl,HeadsetDualArmControl,HeadsetRightControl
from quest.headset_utils import HeadsetFeedback

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Record demonstrations for Isaac Lab environments.")
#parser.add_argument("--task", type=str, default="Isaac-Stack-Cube-Franka-Bimanual-IK-Abs-v0", help="Name of the task.")
parser.add_argument("--task", type=str, default="Isaac-Open-Drawer-Franka-Bimanual-IK-Abs-v0", help="Name of the task.")
parser.add_argument("--teleop_device", type=str, default="meta_quest", help="Device for interacting with environment.")
parser.add_argument(
    "--dataset_file", type=str, default="datasets/dataset.hdf5", help="File path to export recorded demos."
)
parser.add_argument("--step_hz", type=int, default=30, help="Environment stepping rate in Hz.")
parser.add_argument(
    "--num_demos", type=int, default=5, help="Number of demonstrations to record. Set to 0 for infinite."
)
parser.add_argument(
    "--num_success_steps",
    type=int,
    default=10,
    help="Number of continuous steps with task success for concluding a demo as successful. Default is 10.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

if args_cli.teleop_device.lower() == "handtracking":
    vars(args_cli)["experience"] = f'{os.environ["ISAACLAB_PATH"]}/apps/isaaclab.python.xr.openxr.kit'

# launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import contextlib
import gymnasium as gym
import time
import torch

import omni.log

from isaaclab.devices import Se3HandTracking, Se3Keyboard, Se3SpaceMouse
from isaaclab.envs import ViewerCfg
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.envs.ui import ViewportCameraController

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG

# utils tools
to_numpy = lambda x: x.detach().cpu().numpy().squeeze(0).astype(np.float64)
RIGHT_PANDA_BTV_QUAT = [-0.707, 0.0, 0.0, 0.707]  # xyzw
LEFT_PANDA_BTV_QUAT = [0.707, 0.0, 0.0, 0.707]  # xyzw


class RateLimiter:
    """Convenience class for enforcing rates in loops."""

    def __init__(self, hz):
        """
        Args:
            hz (int): frequency to enforce
        """
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.033, self.sleep_duration)

    def sleep(self, env):
        """Attempt to sleep at the specified rate in hz."""
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()

        self.last_time = self.last_time + self.sleep_duration

        # detect time jumping forwards (e.g. loop is too slow)
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


def pre_process_actions(pose: torch.Tensor, gripper_command: bool) -> torch.Tensor:
    """Pre-process actions for the environment."""
    # compute actions based on environment
    if "Reach" in args_cli.task:
        # note: reach is the only one that uses a different action space
        # compute actions
        return pose
    else:
        # resolve gripper command
        gripper_vel = torch.zeros((pose.shape[0], 1), dtype=torch.float, device=pose.device)
        gripper_vel[:] = -1 if gripper_command else 1
        # compute actions
        return torch.concat([pose, gripper_vel], dim=1)
    
def wait_for_user(env, headset:WebRTCHeadset, message="Align and hold A to start the episode."):


    # press gripper to start data collection
    # disable torque for only gripper joint of master robot to allow user movement

    headset_control = HeadsetDualArmControl(right_btv_quat=RIGHT_PANDA_BTV_QUAT,
                                            left_btv_quat=LEFT_PANDA_BTV_QUAT,
                                            wxyz=True)
    feedback = HeadsetFeedback()
    headset_control.reset()
    # wait for user to start the episode
    print("Waiting for user to start the episode...")
    # 获取相对于world夹爪位置姿态
    current_ee_pos_right_w = to_numpy(env.obs_buf["policy"]["eef_pos_right_w"])
    current_ee_quat_right_w = to_numpy(env.obs_buf["policy"]["eef_quat_right_w"])  # wxyz
    current_ee_pose_right_w = np.concatenate([current_ee_pos_right_w, current_ee_quat_right_w])
    # 获取相对于base夹爪位置姿态
    current_ee_pos_right_b = to_numpy(env.obs_buf["policy"]["eef_pos_right_b"])
    current_ee_quat_right_b = to_numpy(env.obs_buf["policy"]["eef_quat_right_b"])
    current_ee_pose_right_b = np.concatenate([current_ee_pos_right_b, current_ee_quat_right_b])

    current_ee_pos_left_b = to_numpy(env.obs_buf["policy"]["eef_pos_left_b"])
    current_ee_quat_left_b = to_numpy(env.obs_buf["policy"]["eef_quat_left_b"])
    current_ee_pose_left_b = np.concatenate([current_ee_pos_left_b, current_ee_quat_left_b])

    
    while True:
        start_time = time.time()
        # 绝对位置姿态
        abs_pose_left, gripper_command_left = current_ee_pose_left_b, False
        abs_pose_left = abs_pose_left.astype("float32")

        abs_pose_right, gripper_command_right = current_ee_pose_right_b, False
        abs_pose_right = abs_pose_right.astype("float32")
        # convert to torch
        abs_pose_left = torch.tensor(abs_pose_left, device=env.device).repeat(env.num_envs, 1)
        abs_pose_right = torch.tensor(abs_pose_right, device=env.device).repeat(env.num_envs, 1)
        # pre-process actions
        actions_left = pre_process_actions(abs_pose_left, gripper_command_left)
        actions_right = pre_process_actions(abs_pose_right, gripper_command_right)

        actions= torch.concatenate((actions_left,actions_right),dim=-1)

        # send initial image to headset
        #zed_img = ts["images"]["zed_cam"]
        headset.send_images(left_image=to_numpy(env.obs_buf["policy"]["zed_left"]).astype(np.uint8),
                            right_image=to_numpy(env.obs_buf["policy"]["zed_right"]).astype(np.uint8))
        env.step(actions)


        #ee_marker.visualize(current_pose_w[:, 0:3], current_pose_w[:, 3:7])

        headset_data = headset.receive_data()
        if headset_data is not None:
            
            # break if the user holds the right button
            if headset_data.r_button_one == True and feedback.head_out_of_sync == False and \
                feedback.left_out_of_sync == False and feedback.right_out_of_sync == False:
                headset_control.start(headset_data, current_ee_pose_right_b, current_ee_pose_left_b)
                break


        feedback.info = message
        headset.send_feedback(feedback)

    print(f'Started!')

    return headset_control



def main():
    """Collect demonstrations from the environment using teleop interfaces."""

    # if handtracking is selected, rate limiting is achieved via OpenXR
    if args_cli.teleop_device.lower() == "handtracking":
        rate_limiter = None
    else:
        rate_limiter = RateLimiter(args_cli.step_hz)

    # get directory path and file name (without extension) from cli arguments
    output_dir = os.path.dirname(args_cli.dataset_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]

    # create directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.env_name = args_cli.task

    # extract success checking function to invoke in the main loop
    success_term = None
    if hasattr(env_cfg.terminations, "success"):
        success_term = env_cfg.terminations.success
        env_cfg.terminations.success = None
    else:
        omni.log.warn(
            "No success termination term was found in the environment."
            " Will not be able to mark recorded demos as successful."
        )

    # modify configuration such that the environment runs indefinitely until
    # the goal is reached or other termination conditions are met
    env_cfg.terminations.time_out = None

    env_cfg.observations.policy.concatenate_terms = False

    env_cfg.recorders: ActionStateRecorderManagerCfg = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name

    env_cfg.viewer.eye = (-2.0, -2.0, 3.0)     # 摄像机的位置
    env_cfg.viewer.lookat = (4.0, 2.0, 0.0)  # 摄像机看的目标位置
    env_cfg.viewer.origin_type = "world"     # 相对于世界坐标系

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    # add teleoperation key for reset current recording instance
    should_reset_recording_instance = False

    def reset_recording_instance():
        nonlocal should_reset_recording_instance
        should_reset_recording_instance = True

    # quest控制器
    headset = WebRTCHeadset()
    headset.run_in_thread()

    # simulate environment -- run everything in inference mode
    current_recorded_demo_count = 0
    success_step_count = 0
    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        # episode 循环
        while True:
            # demo数量足够，将跳出最外层循环
            if args_cli.num_demos > 0 and env.recorder_manager.exported_successful_episode_count >= args_cli.num_demos:
                print(f"All {args_cli.num_demos} demonstrations recorded. Exiting the app.")
                break
            # check that simulation is stopped or not
            if env.sim.is_stopped():
                break

            env.reset()
            should_reset_recording_instance = False
            success_step_count = 0 # 清零回合成功保持步数计数器
            abort_step_count = 0

            # wait user connection
            headset_control = wait_for_user(env, headset)
            feedback = HeadsetFeedback()
            # 在用户准备好之后再执行demo记录
            env.recorder_manager.reset()
            
            # step循环
            while True:
                # 读取hmd 控制量
                headset_data = headset.receive_data()

                if headset_data is not None:
                    # 左手当前姿态
                    current_ee_pos_left_b = to_numpy(env.obs_buf["policy"]["eef_pos_left_b"])
                    current_ee_quat_left_b = to_numpy(env.obs_buf["policy"]["eef_quat_left_b"])
                    current_ee_pose_left_b = np.concatenate([current_ee_pos_left_b, current_ee_quat_left_b])
                    # 右手当前姿态
                    current_ee_pos_right_b = to_numpy(env.obs_buf["policy"]["eef_pos_right_b"])
                    current_ee_quat_right_b = to_numpy(env.obs_buf["policy"]["eef_quat_right_b"])
                    current_ee_pose_right_b = np.concatenate([current_ee_pos_right_b, current_ee_quat_right_b])

                    action, feedback = headset_control.run(
                        headset_data, 
                        current_ee_pose_right_b, # panda
                        current_ee_pose_left_b # cr5 占位
                    )

                    # get command
                    abs_pose_left, gripper_command_left = action[-8:-1], headset_data.l_hand_trigger
                    abs_pose_left = abs_pose_left.astype("float32")

                    abs_pose_right, gripper_command_right = action[:7], headset_data.r_hand_trigger
                    abs_pose_right = abs_pose_right.astype("float32")
                    # convert to torch
                    abs_pose_left = torch.tensor(abs_pose_left, device=env.device).repeat(env.num_envs, 1)
                    abs_pose_right = torch.tensor(abs_pose_right, device=env.device).repeat(env.num_envs, 1)
                    # pre-process actions
                    actions_left = pre_process_actions(abs_pose_left, gripper_command_left)
                    actions_right = pre_process_actions(abs_pose_right, gripper_command_right)

                    feedback.info = ""
                    headset.send_feedback(feedback)

                    headset.send_images(left_image=to_numpy(env.obs_buf["policy"]["zed_left"]).astype(np.uint8),
                                        right_image=to_numpy(env.obs_buf["policy"]["zed_right"]).astype(np.uint8))
                    actions= torch.concatenate((actions_left,actions_right),dim=-1)
                    # apply actions
                    env.step(actions)

                    # 在每一步都判断是否成功
                    if success_term is not None:
                        if bool(success_term.func(env, **success_term.params)[0]):
                            success_step_count += 1
                            if success_step_count >= args_cli.num_success_steps:
                                env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
                                env.recorder_manager.set_success_to_episodes(
                                    [0], torch.tensor([[True]], dtype=torch.bool, device=env.device)
                                )
                                env.recorder_manager.export_episodes([0])
                                should_reset_recording_instance = True
                        else:
                            success_step_count = 0
                    
                    # 检测右手按钮A按钮是否长按
                    if headset_data.r_button_one == True:
                        abort_step_count += 1
                        if abort_step_count >=10:
                            feedback.info = "Relase button A to abort current demonstration."
                            headset.send_feedback(feedback)
                            print(f"Relase button A to abort current demonstration.")
                    elif abort_step_count >10:
                            should_reset_recording_instance = True
                            env.recorder_manager.reset()
                            
                    else:
                        abort_step_count = 0

                # print out the current demo count if it has changed
                if env.recorder_manager.exported_successful_episode_count > current_recorded_demo_count:
                    current_recorded_demo_count = env.recorder_manager.exported_successful_episode_count
                    print(f"Recorded {current_recorded_demo_count} successful demonstrations.")


                # check that simulation is stopped or not
                if env.sim.is_stopped():
                    break

                if rate_limiter:
                    rate_limiter.sleep(env)
                # 根据重置标志位判断是否应该跳出当前回合
                if should_reset_recording_instance:
                    break


    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
