# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Union, Sequence

from isaaclab.assets import Articulation, RigidObject, RigidObjectCollection
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer, FrameTransformerData

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def _as_seq(x: Union[SceneEntityCfg, Sequence[SceneEntityCfg]]) -> Sequence[SceneEntityCfg]:
    return (x,) if isinstance(x, SceneEntityCfg) else x

def rel_ee_drawer_distance(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The distance between the end-effector and the object."""
    ee_tf_data: FrameTransformerData = env.scene["ee_frame_right"].data
    drawer_tf_data: FrameTransformerData = env.scene["table_frame"].data
    return drawer_tf_data.target_pos_w[..., 0, :] - ee_tf_data.target_pos_w[..., 0, :]

def gripper_pos(
    env: ManagerBasedRLEnv,
    robot_cfgs: Union[SceneEntityCfg, Sequence[SceneEntityCfg]] = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    获取一个或多个机器人的 gripper finger 关节位置。

    - 单个 robot: 返回 (B, 2)
    - 多个 robot: 返回 (B, 2*N)，按传入顺序拼接
    """
    cfgs = _as_seq(robot_cfgs)
    chunks = []
    for cfg in cfgs:
        robot: Articulation = env.scene[cfg.name]
        f1 = robot.data.joint_pos[:, -1].clone().unsqueeze(1)
        f2 = -1 * robot.data.joint_pos[:, -2].clone().unsqueeze(1)
        chunks.append(torch.cat((f1, f2), dim=1))  # (B,2)
    return torch.cat(chunks, dim=-1)  # (B,2*N)




def ee_frame_pos_b(
    env: ManagerBasedRLEnv,
    ee_frames: Union[SceneEntityCfg, Sequence[SceneEntityCfg]] = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """
    基于 base/source 坐标系的 EE 位置。
    单个返回形状 (B, 3)，多个返回 (B, 3*N)，按传入顺序拼接。
    """
    frames = _as_seq(ee_frames)
    chunks = []
    for cfg in frames:
        ee: FrameTransformer = env.scene[cfg.name]
        # target 索引按你当前的数据结构保持 [:, 0, :]
        chunks.append(ee.data.target_pos_source[:, 0, :])  # (B, 3)
    return torch.cat(chunks, dim=-1)  # (B, 3*N)


def ee_frame_quat_b(
    env: ManagerBasedRLEnv,
    ee_frames: Union[SceneEntityCfg, Sequence[SceneEntityCfg]] = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """
    基于 base/source 坐标系的 EE 四元数 (wxyz)。
    单个返回 (B, 4)，多个返回 (B, 4*N)。
    """
    frames = _as_seq(ee_frames)
    chunks = []
    for cfg in frames:
        ee: FrameTransformer = env.scene[cfg.name]
        chunks.append(ee.data.target_quat_source[:, 0, :])  # (B, 4)
    return torch.cat(chunks, dim=-1)  # (B, 4*N)


def ee_frame_pos_w(
    env: ManagerBasedRLEnv,
    ee_frames: Union[SceneEntityCfg, Sequence[SceneEntityCfg]] = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """
    基于世界/环境坐标系的 EE 位置（减去 env 原点对齐到每个子环境坐标）。
    单个返回 (B, 3)，多个返回 (B, 3*N)。
    """
    frames = _as_seq(ee_frames)
    env_origins = env.scene.env_origins[:, 0:3]  # (B, 3)
    chunks = []
    for cfg in frames:
        ee: FrameTransformer = env.scene[cfg.name]
        pos_w = ee.data.target_pos_w[:, 0, :] - env_origins  # (B, 3)
        chunks.append(pos_w)
    return torch.cat(chunks, dim=-1)  # (B, 3*N)


def ee_frame_quat_w(
    env: ManagerBasedRLEnv,
    ee_frames: Union[SceneEntityCfg, Sequence[SceneEntityCfg]] = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """
    基于世界/环境坐标系的 EE 四元数 (wxyz)。
    单个返回 (B, 4)，多个返回 (B, 4*N)。
    """
    frames = _as_seq(ee_frames)
    chunks = []
    for cfg in frames:
        ee: FrameTransformer = env.scene[cfg.name]
        chunks.append(ee.data.target_quat_w[:, 0, :])  # (B, 4)
    return torch.cat(chunks, dim=-1)  # (B, 4*N)
 



def handle_grasped(
    env: ManagerBasedRLEnv,
    dist_threshold: float = 0.05,
) -> torch.Tensor:
    """Check if an object is grasped by the specified robot."""

    ee_tf_data: FrameTransformerData = env.scene["ee_frame_right"].data
    drawer_tf_data: FrameTransformerData = env.scene["table_frame"].data
    dist =torch.norm(drawer_tf_data.target_pos_w[..., 0, :] - ee_tf_data.target_pos_w[..., 0, :], dim=1)

    grasped = dist < dist_threshold

    return grasped


def drawer_dragged(
    env: ManagerBasedRLEnv,
    dist_threshold: float = 0.1,
) -> torch.Tensor:
    drawer_tf_data: FrameTransformerData = env.scene["table_frame"].data
    # 查看偏差
    dist_x = torch.norm(drawer_tf_data.target_pos_w[..., 0, [0]] - drawer_tf_data.target_pos_w[..., 1, [0]],dim=0)
    # print(dist_x,dist_y,dist_z)
    dragged = torch.logical_and(dist_x > dist_threshold, dist_x > dist_threshold)

    return dragged



