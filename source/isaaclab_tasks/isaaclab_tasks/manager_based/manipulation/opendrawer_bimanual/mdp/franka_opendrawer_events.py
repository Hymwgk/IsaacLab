# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import math
import random
import torch
from typing import TYPE_CHECKING, Sequence, Union
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, AssetBase
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema
import omni.usd


def _as_seq(x):
    return (x,) if not isinstance(x, (list, tuple)) else x



def set_default_joint_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    default_pose: Union[torch.Tensor, Sequence[torch.Tensor]] | None,
    asset_cfg: Union[SceneEntityCfg, Sequence[SceneEntityCfg]] = SceneEntityCfg("robot"),
):
    """
    """
    cfgs = _as_seq(asset_cfg)

    if default_pose is not None:
        poses = _as_seq(default_pose)
        for pose, cfg  in zip(poses, cfgs):
            asset: Articulation = env.scene[cfg.name]
            pose = torch.as_tensor(pose, device=env.device).view(1, -1)  # (1, DoF)
            dof = asset.data.default_joint_pos.shape[-1]
            assert pose.shape[-1] == dof 
            asset.data.default_joint_pos = (pose.repeat(env.num_envs, 1))
    else:
        for cfg  in cfgs:
            asset: Articulation = env.scene[cfg.name]
            pose = asset.data.default_joint_pos  # (1, DoF)
            dof = asset.data.default_joint_pos.shape[-1]
            assert pose.shape[-1] == dof 
            asset.data.default_joint_pos = (pose.repeat(env.num_envs, 1))

def reset_to_default_joint_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    default_pose: Union[torch.Tensor, Sequence[torch.Tensor]] | None,
    asset_cfg: Union[SceneEntityCfg, Sequence[SceneEntityCfg]] = SceneEntityCfg("robot"),
):
    """
    """
    cfgs = _as_seq(asset_cfg)

    if default_pose is not None:
        poses = _as_seq(default_pose)
        for pose, cfg  in zip(poses, cfgs):
            asset: Articulation = env.scene[cfg.name]
            pose = torch.as_tensor(pose, device=env.device).view(1, -1)  # (1, DoF)
            dof = asset.data.default_joint_pos.shape[-1]
            assert pose.shape[-1] == dof 
            asset.write_joint_position_to_sim(pose.repeat(env.num_envs, 1))
    else:
        for cfg  in cfgs:
            asset: Articulation = env.scene[cfg.name]
            pose = asset.data.default_joint_pos  # (1, DoF)
            dof = asset.data.default_joint_pos.shape[-1]
            assert pose.shape[-1] == dof 
            asset.write_joint_position_to_sim(pose.repeat(env.num_envs, 1))

def randomize_joint_by_gaussian_offset(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    mean: float,
    std: float,
    asset_cfg: Union[SceneEntityCfg, Sequence[SceneEntityCfg]] = SceneEntityCfg("robot"),
    keep_last_two_as_gripper: bool = True,
):
    """对一个或多个机器人的关节状态在 default 附近加高斯噪声并写回仿真。
    - 对每个 asset：
        * q := default_joint_pos[env_ids] + N(mean, std)
        * clamp 到 soft_joint_pos_limits
        * 可选：最后两个关节（通常是 gripper）不加噪声
        * 速度用 default_joint_vel[env_ids]
        * 同步 set_position/velocity_target + write_joint_state_to_sim
    """
    cfgs = _as_seq(asset_cfg)

    for cfg in cfgs:
        asset: Articulation = env.scene[cfg.name]

        # 取默认值
        q0 = asset.data.default_joint_pos[env_ids].clone()          # (E, DoF)
        qd0 = asset.data.default_joint_vel[env_ids].clone()         # (E, DoF)

        # 加噪
        noise = math_utils.sample_gaussian(mean, std, q0.shape, q0.device)
        q = q0 + noise

        # 限幅
        limits = asset.data.soft_joint_pos_limits[env_ids]          # (E, DoF, 2)
        q = q.clamp_(limits[..., 0], limits[..., 1])

        # 夹爪关节保持默认（假设为最后两个 DoF；若不是请改成显式索引）
        if keep_last_two_as_gripper and q.shape[-1] >= 2:
            q[:, -2:] = q0[:, -2:]

        # 写入仿真 & 目标
        asset.set_joint_position_target(q, env_ids=env_ids)
        asset.set_joint_velocity_target(qd0, env_ids=env_ids)
        asset.write_joint_state_to_sim(q, qd0, env_ids=env_ids)


def randomize_scene_lighting_domelight(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    intensity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("light"),
):
    asset: AssetBase = env.scene[asset_cfg.name]
    light_prim = asset.prims[0]

    # Sample new light intensity
    new_intensity = random.uniform(intensity_range[0], intensity_range[1])

    # Set light intensity to light prim
    intensity_attr = light_prim.GetAttribute("inputs:intensity")
    intensity_attr.Set(new_intensity)


def sample_object_poses(
    num_objects: int,
    min_separation: float = 0.0,
    pose_range: dict[str, tuple[float, float]] = {},
    max_sample_tries: int = 5000,
):
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    pose_list = []

    for i in range(num_objects):
        for j in range(max_sample_tries):
            sample = [random.uniform(range[0], range[1]) for range in range_list]

            # Accept pose if it is the first one, or if reached max num tries
            if len(pose_list) == 0 or j == max_sample_tries - 1:
                pose_list.append(sample)
                break

            # Check if pose of object is sufficiently far away from all other objects
            separation_check = [math.dist(sample[:3], pose[:3]) > min_separation for pose in pose_list]
            if False not in separation_check:
                pose_list.append(sample)
                break

    return pose_list


def randomize_object_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfgs: list[SceneEntityCfg],
    min_separation: float = 0.0,
    pose_range: dict[str, tuple[float, float]] = {},
    max_sample_tries: int = 5000,
):
    if env_ids is None:
        return

    # Randomize poses in each environment independently
    for cur_env in env_ids.tolist():
        pose_list = sample_object_poses(
            num_objects=len(asset_cfgs),
            min_separation=min_separation,
            pose_range=pose_range,
            max_sample_tries=max_sample_tries,
        )

        # Randomize pose for each object
        for i in range(len(asset_cfgs)):
            asset_cfg = asset_cfgs[i]
            asset = env.scene[asset_cfg.name]

            # Write pose to simulation
            pose_tensor = torch.tensor([pose_list[i]], device=env.device)
            positions = pose_tensor[:, 0:3] + env.scene.env_origins[cur_env, 0:3]
            orientations = math_utils.quat_from_euler_xyz(pose_tensor[:, 3], pose_tensor[:, 4], pose_tensor[:, 5])
            asset.write_root_pose_to_sim(
                torch.cat([positions, orientations], dim=-1), env_ids=torch.tensor([cur_env], device=env.device)
            )
            asset.write_root_velocity_to_sim(
                torch.zeros(1, 6, device=env.device), env_ids=torch.tensor([cur_env], device=env.device)
            )



