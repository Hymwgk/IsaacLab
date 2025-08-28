# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors import TiledCameraCfg


from isaaclab.markers import VisualizationMarkers


from isaaclab.sensors.frame_transformer import OffsetCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from . import mdp

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip


FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy()
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.10, 0.10, 0.10)


##
# Scene definition
##


@configclass
class TableSceneCfg(InteractiveSceneCfg):
    """Configuration for the table scene with a robot and a table.

    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the robot and end-effector frames
    """

    # Markers
    # ee_marker = VisualizationMarkers(FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/ee_current"))
    # goal_marker = VisualizationMarkers(FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/ee_goal"))


    # robots, Will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # End-effector, Will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING


    # add camera to the scene
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(-7.0, 0.0, 3.0), rot=(0.9945, 0.0, 0.1045, 0.0), convention="world"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        ),
        width=100,
        height=100,
    )



    table = ArticulationCfg(
        # 设置物体的路径，就是在场景树中相对于环境的路径
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Sektion_Table/sektion_table_instanceable.usd",
            activate_contact_sensors=False,
        ),
        # 设置初始状态
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.8, 0, 0.4),
            rot=(0.0, 0.0, 0.0, 1.0),
            joint_pos={
                "door_left_joint": 0.0,
                "door_right_joint": 0.0,
                "drawer_bottom_joint": 0.0,
                "drawer_top_joint": 0.0,
            },
        ),
        # 设置内部关节限制
        actuators={
            "drawers": ImplicitActuatorCfg(
                joint_names_expr=["drawer_top_joint", "drawer_bottom_joint"],
                effort_limit=10.0, # 这里我偷懒，降低了抽屉的容易拉开程度，并不太符合物理标准 原87
                velocity_limit=200.0, # 原100
                stiffness=0.0, # 原10
                damping=0.0, # 原1.0
            ),
            "doors": ImplicitActuatorCfg(
                joint_names_expr=["door_left_joint", "door_right_joint"],
                effort_limit=87.0,
                velocity_limit=100.0,
                stiffness=10.0,
                damping=2.5,
            ),
        },
    )





    # 获取相对于柜子的变动的一些位置姿态（变换关系）
    table_frame = FrameTransformerCfg(
        # 设置父坐标系的xform
        prim_path="{ENV_REGEX_NS}/Table",
        debug_vis=False,
        visualizer_cfg= FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/TableFrameTransformer"),
        target_frames=[
            # 手柄坐标系的名字还是“drawer_handle_top”
            FrameTransformerCfg.FrameCfg(
                # 手柄坐标系在哪里？它要固定附着在场景中已经存在的参考系，两者“相对位姿固定”
                # 在这里，用"{ENV_REGEX_NS}/Table/drawer_handle_top"这个xform来作为附着对象
                prim_path="{ENV_REGEX_NS}/Table/drawer_handle_top",
                # 手柄的坐标系的名称，它起了一个与prim_path中相同的名字
                name="drawer_handle_top",
                offset=OffsetCfg(
                    pos=(0.305, 0.0, 0.01),
                    rot=(0.5, 0.5, -0.5, -0.5),  # align with end-effector frame
                ),
            ),
        ],
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(),
        spawn=sim_utils.GroundPlaneCfg(),
        collision_group=-1,
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action: ActionTerm = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        # 声明rgb观测
        rgb_image = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("tiled_camera"), 
                                                    "data_type": "rgb"})
        # # 声明depth图观测
        # depth_image = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("tiled_camera"),
        #                                               "data_type": "distance_to_camera"})
        
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        table_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("table", joint_names=["drawer_top_joint"])},
        )
        table_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("table", joint_names=["drawer_top_joint"])},
        )
        rel_ee_drawer_distance = ObsTerm(func=mdp.rel_ee_drawer_distance)
        actions = ObsTerm(func=mdp.last_action)
        
        # waypoint states
        waypoint_states: ObsTerm = MISSING
        # 获取当前的夹爪位置姿态
        ee_pos = ObsTerm(func=mdp.ee_pos)
        ee_quat = ObsTerm(func=mdp.ee_quat)
        # 补充一个gripper的观测，用来判断当前的状态, state = 1 打开，否则是关闭状态
        gripper_state = ObsTerm(func=mdp.gripper_state)

        def __post_init__(self):
            self.enable_corruption = True
            # 这里我改动为False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 1.25),
            "dynamic_friction_range": (0.8, 1.25),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )

    table_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("table", body_names="drawer_handle_top"),
            "static_friction_range": (1.0, 1.25),
            "dynamic_friction_range": (1.25, 1.5),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.1, 0.1),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # 1. Approach the handle
    approach_ee_handle = RewTerm(func=mdp.approach_ee_handle, weight=2.0, params={"threshold": 0.2})
    align_ee_handle = RewTerm(func=mdp.align_ee_handle, weight=0.5)

    # 2. Grasp the handle
    approach_gripper_handle = RewTerm(func=mdp.approach_gripper_handle, weight=5.0, params={"offset": MISSING})
    align_grasp_around_handle = RewTerm(func=mdp.align_grasp_around_handle, weight=0.125)
    grasp_handle = RewTerm(
        func=mdp.grasp_handle,
        weight=0.5,
        params={
            "threshold": 0.03,
            "open_joint_pos": MISSING,
            "asset_cfg": SceneEntityCfg("robot", joint_names=MISSING),
        },
    )

    # 3. Open the drawer
    open_drawer_bonus = RewTerm(
        func=mdp.open_drawer_bonus,
        weight=7.5,
        params={"asset_cfg": SceneEntityCfg("table", joint_names=["drawer_top_joint"])},
    )
    multi_stage_open_drawer = RewTerm(
        func=mdp.multi_stage_open_drawer,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("table", joint_names=["drawer_top_joint"])},
    )

    # 4. Penalize actions for cosmetic reasons
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1e-2)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.0001)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # 声明一个简单的成功判断函数，具体的自定义success函数写在了observation.py中
    success = DoneTerm(func=mdp.success)


##
# Environment configuration
##


@configclass
class TableEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the table environment."""

    # Scene settings
    scene: TableSceneCfg = TableSceneCfg(num_envs=4096, env_spacing=2.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 1
        self.episode_length_s = 8.0
        self.viewer.eye = (-2.0, 2.0, 2.0)
        self.viewer.lookat = (0.8, 0.0, 0.5)
        # simulation settings
        self.sim.dt = 1 / 60  # 60Hz
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625
