# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR,ISAACLAB_NUCLEUS_DIR
from isaaclab.managers import SceneEntityCfg

from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.sensors import CameraCfg, FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils.math import quat_from_euler_xyz, subtract_frame_transforms
from isaaclab.managers import ObservationTermCfg as ObsTerm

from . import opendrawer_joint_pos_env_cfg
from ... import mdp
##
# Pre-defined configs
##
from isaaclab_assets.robots.franka import (FRANKA_PANDA_HIGH_PD_CFG,FRANKA_PANDA_TACT_HIGH_PD_CFG)  # isort: skip


@configclass
class FrankaOpenDrawerEnvCfg(opendrawer_joint_pos_env_cfg.FrankaOpenDrawerEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Franka as robot
        # We switch here to a stiffer PD controller for IK tracking to be better.
        # zed相机是一个带有物理性质的物体，不能随便次级挂载到某个xform下
        self.scene.robot_right = FRANKA_PANDA_TACT_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/panda_right",
                                init_state=ArticulationCfg.InitialStateCfg(pos=[0, -0.05, 1.6],rot=[0.707107, 0.707107, 0.0, 0.0],))#wxyz  
        self.scene.robot_left = FRANKA_PANDA_TACT_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/panda_left",
                                init_state=ArticulationCfg.InitialStateCfg(pos=[0, 0.05, 1.6],rot=[0.707107, -0.707107, 0.0, 0.0],))#wxyz    
        
        # Set zed cameras
        # zed相机是一个带有物理性质的物体，不能随便次级挂载到某个xform下，比如这里的panda_handa坐标系下
        # 需要用fixed joint来固定连接两个物体才行，那样，就相当于把zed相机和panda机械臂安装在一起了，因此要导入 FRANKA_PANDA_ZED_HIGH_PD_CFG

        self.scene.zed_left = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Stand/ZED_X/base_link/ZED_X/CameraLeft",
            update_period=0.0333,
            height=480,
            width=640,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=None,
        )

        self.scene.zed_right = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Stand/ZED_X/base_link/ZED_X/CameraRight",
            update_period=0.0333,
            height=480,
            width=640,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=None,
        )

        self.scene.zed_middle = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Stand/ZED_X/base_link/ZED_X/CameraMiddle",
            update_period=0.0333,
            height=480,
            width=640,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=None,
        )



        self.scene.tact_right_right = CameraCfg(
            prim_path="{ENV_REGEX_NS}/panda_right/digit_right/finger_tip_body/tact_cam",
            update_period=0.0333,
            height=320,
            width=240,
            data_types=["rgb"],
            spawn=None,
        )
        self.scene.tact_right_left = CameraCfg(
            prim_path="{ENV_REGEX_NS}/panda_right/digit_left/finger_tip_body/tact_cam",
            update_period=0.0333,
            height=320,
            width=240,
            data_types=["rgb"],
            spawn=None,
        )
        self.scene.tact_left_right = CameraCfg(
            prim_path="{ENV_REGEX_NS}/panda_left/digit_right/finger_tip_body/tact_cam",
            update_period=0.0333,
            height=320,
            width=240,
            data_types=["rgb"],
            spawn=None,
        )
        self.scene.tact_left_left = CameraCfg(
            prim_path="{ENV_REGEX_NS}/panda_left/digit_left/finger_tip_body/tact_cam",
            update_period=0.0333,
            height=320,
            width=240,
            data_types=["rgb"],
            spawn=None,
        )



        self.scene.camera_light_l_l = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/panda_left/digit_left/finger_tip_body/L",
            spawn=None)
        self.scene.camera_light_l_r = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/panda_left/digit_right/finger_tip_body/L",
            spawn=None)

        self.scene.camera_light_r_l = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/panda_right/digit_left/finger_tip_body/L",
            spawn=None)
        self.scene.camera_light_r_r = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/panda_right/digit_right/finger_tip_body/L",
            spawn=None)



        self.observations.policy.tact_l_l = ObsTerm(
            func=mdp.image, params={"sensor_cfg": SceneEntityCfg("tact_left_left"), "data_type": "rgb", "normalize": False}
        )
        self.observations.policy.tact_l_r = ObsTerm(
            func=mdp.image, params={"sensor_cfg": SceneEntityCfg("tact_left_right"), "data_type": "rgb", "normalize": False}
        )
        self.observations.policy.tact_r_l = ObsTerm(
            func=mdp.image, params={"sensor_cfg": SceneEntityCfg("tact_right_left"), "data_type": "rgb", "normalize": False}
        )
        self.observations.policy.tact_r_r = ObsTerm(
            func=mdp.image, params={"sensor_cfg": SceneEntityCfg("tact_right_right"), "data_type": "rgb", "normalize": False}
        )        


        # Set actions for the specific robot type (franka)
        self.actions.arm_action_left = DifferentialInverseKinematicsActionCfg(
            asset_name="robot_left",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            #scale=1.0,    # 这里，如果是输入绝对动作，必须是1.0 ，或者直接删除，不能有折扣
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        )

        self.actions.arm_action_right = DifferentialInverseKinematicsActionCfg(
            asset_name="robot_right",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            #scale=1.0,    # 这里，如果是输入绝对动作，必须是1.0 ，或者直接删除，不能有折扣
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        )

