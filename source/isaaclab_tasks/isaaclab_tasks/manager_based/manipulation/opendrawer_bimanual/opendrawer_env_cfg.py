# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.sensors.frame_transformer import OffsetCfg

from . import mdp


##
# Scene definition
##
@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and a object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """

    # robots: will be populated by agent env cfg
    robot_left: ArticulationCfg = MISSING
    robot_right: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_frame_left: FrameTransformerCfg = MISSING
    ee_frame_right: FrameTransformerCfg = MISSING
    
    # # Cameras
    wrist_cam_left: CameraCfg = MISSING
    wrist_cam_right: CameraCfg = MISSING
    
    table_cam: CameraCfg = MISSING


    # Table
    table = ArticulationCfg(
        # 设置物体的路径，就是在场景树中相对于环境的路径
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/work_table.usdc",
                         scale=(0.5, 0.5, 0.5)),
        # 设置初始状态
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.5, 0.0, 0), rot=(0.707, 0.0, 0.0, -0.707),
            joint_pos={
                "door_left_joint": 0.0,
                "door_right_joint": 0.0,
                "drawer_left_joint": -0.3,
                "drawer_right_joint": -0.3,
            },
        ),
        # 设置内部关节限制
        actuators={
            "drawers": ImplicitActuatorCfg(
                joint_names_expr=["drawer_left_joint", "drawer_right_joint"],
                effort_limit=87.0, # 这里我偷懒，降低了抽屉的容易拉开程度，并不太符合物理标准 原87
                velocity_limit=100.0, # 原100
                stiffness=10.0, # 原10
                damping=10.0, # 原1.0
                
            ),
            "doors": ImplicitActuatorCfg(
                joint_names_expr=["door_left_joint", "door_right_joint"],
                effort_limit=87.0,
                velocity_limit=100.0,
                stiffness=10.0,
                damping=10.5,
            ),
        },
    )




    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, 0)),  # 地面
        spawn=GroundPlaneCfg(),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


    # zed双目相机  这种CameraCfg形式的声明  
    # 无论书写顺序，一定会提前于AssetBaseCfg先初始化 这导致会找不到物体
    # 如果相机挂载了其他物体上，就不能这样子声明，要直接在后续实例化即可
    # zed_left: CameraCfg = MISSING
    # zed_right: CameraCfg = MISSING

    # Stand
    stand = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Stand",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 1.781), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/Stand/stand_zed.usd",
                         scale=(1.0, 1.0, 1.0)),
    )


    # 获取相对于桌子的变动的一些位置姿态（变换关系）
    table_frame = FrameTransformerCfg(
        # 设置父坐标系的xform
        prim_path="{ENV_REGEX_NS}/Table/back_frame",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                # 手柄坐标系在哪里？它要固定附着在场景中已经存在的参考系，两者“相对位姿固定”
                prim_path="{ENV_REGEX_NS}/Table/drawer_right",
                name="drawer_handle_right",
                offset=OffsetCfg(
                    pos=(0.0, -0.37821, -0.05387),
                    rot=(1.0, 0.0, 0.0, 0.0), # align with end-effector frame
                ),
            ),

            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Table/back_frame",
                name="drawer_reference",
            ),
        ],
    )


##
# MDP settings
##
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    arm_action_left: mdp.JointPositionActionCfg = MISSING
    gripper_action_left: mdp.BinaryJointPositionActionCfg = MISSING

    arm_action_right: mdp.JointPositionActionCfg = MISSING
    gripper_action_right: mdp.BinaryJointPositionActionCfg = MISSING

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        actions = ObsTerm(func=mdp.last_action)

        joint_pos_left = ObsTerm(func=mdp.joint_pos,params={"asset_cfg": SceneEntityCfg("robot_left")})
        joint_pos_right = ObsTerm(func=mdp.joint_pos,params={"asset_cfg": SceneEntityCfg("robot_right")})


        joint_pos_left_rel = ObsTerm(func=mdp.joint_pos_rel,params={"asset_cfg": SceneEntityCfg("robot_left")})
        joint_pos_right_rel = ObsTerm(func=mdp.joint_pos_rel,params={"asset_cfg": SceneEntityCfg("robot_right")})

        joint_vel_left_rel = ObsTerm(func=mdp.joint_vel_rel,params={"asset_cfg": SceneEntityCfg("robot_left")})
        joint_vel_right_rel = ObsTerm(func=mdp.joint_vel_rel,params={"asset_cfg": SceneEntityCfg("robot_right")})
        
        gripper_left_pos = ObsTerm(func=mdp.gripper_pos,params={"robot_cfgs": SceneEntityCfg("robot_left")})
        gripper_right_pos = ObsTerm(func=mdp.gripper_pos,params={"robot_cfgs": SceneEntityCfg("robot_right")})

        # 世界参考系下的eef 位置姿态
        eef_pos_left_w = ObsTerm(func=mdp.ee_frame_pos_w,params={"ee_frames": SceneEntityCfg("ee_frame_left")})
        eef_quat_left_w = ObsTerm(func=mdp.ee_frame_quat_w,params={"ee_frames": SceneEntityCfg("ee_frame_left")})
        eef_pos_right_w = ObsTerm(func=mdp.ee_frame_pos_w,params={"ee_frames": SceneEntityCfg("ee_frame_right")})
        eef_quat_right_w = ObsTerm(func=mdp.ee_frame_quat_w,params={"ee_frames": SceneEntityCfg("ee_frame_right")})


        # base参考系下的eef 位置姿态
        eef_pos_left_b = ObsTerm(func=mdp.ee_frame_pos_b,params={"ee_frames": SceneEntityCfg("ee_frame_left")})
        eef_quat_left_b = ObsTerm(func=mdp.ee_frame_quat_b,params={"ee_frames": SceneEntityCfg("ee_frame_left")})
        eef_pos_right_b = ObsTerm(func=mdp.ee_frame_pos_b,params={"ee_frames": SceneEntityCfg("ee_frame_right")})
        eef_quat_right_b = ObsTerm(func=mdp.ee_frame_quat_b,params={"ee_frames": SceneEntityCfg("ee_frame_right")})
 
        # 如果想要在hdf5数据集文件中保存图像观测；就必须放在   PolicyCfg中
        wrist_cam_left = ObsTerm(
            func=mdp.image, params={"sensor_cfg": SceneEntityCfg("wrist_cam_left"), "data_type": "rgb", "normalize": False}
        )
        wrist_cam_right = ObsTerm(
            func=mdp.image, params={"sensor_cfg": SceneEntityCfg("wrist_cam_right"), "data_type": "rgb", "normalize": False}
        )
        zed_left = ObsTerm(
            func=mdp.image, params={"sensor_cfg": SceneEntityCfg("zed_left"), "data_type": "rgb", "normalize": False}
        )
        zed_right = ObsTerm(
            func=mdp.image, params={"sensor_cfg": SceneEntityCfg("zed_right"), "data_type": "rgb", "normalize": False}
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class RGBCameraPolicyCfg(ObsGroup):
        """Observations for policy group with RGB images."""
        table_cam = ObsTerm(
            func=mdp.image, params={"sensor_cfg": SceneEntityCfg("table_cam"), "data_type": "rgb", "normalize": False}
        )
        def __post_init__(self):
                self.enable_corruption = False
                self.concatenate_terms = False

    @configclass
    class SubtaskCfg(ObsGroup):
        """Observations for subtask group."""

        grasp_1 = ObsTerm(
            func=mdp.handle_grasped,
            params={
                "dist_threshold":  0.05,
            },
        )
        # 子任务设置
        drag_1 = ObsTerm(
            func=mdp.drawer_dragged,
            params={
                "dist_threshold":  0.1,
            },
        )


        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    rgb_camera: RGBCameraPolicyCfg = RGBCameraPolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=mdp.drawer_opened)


@configclass
class OpenDrawerEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the open drawer environment."""

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=False)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()

    actions: ActionsCfg = ActionsCfg()

    # MDP settings
    terminations: TerminationsCfg = TerminationsCfg()

    # Unused managers
    commands = None
    rewards = None
    events = None
    curriculum = None

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 5
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

