# Copyright (c) 2022-2025, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Matterix Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create observation terms.

The functions can be passed to the :class:`matterix.managers.ObservationTermCfg` object to enable
the observation introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import euler_xyz_from_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def force_data(env: ManagerBasedRLEnv, asset_name: str):
    sensor_name = f"contact_sensor_{asset_name}"
    return env.scene[sensor_name].data.net_forces_w


def ee_world_pos(env: ManagerBasedEnv, asset_name: str) -> torch.Tensor:
    ee_frame_name = f"ee_frame_{asset_name}"
    ee_frame: FrameTransformer = env.scene[ee_frame_name]
    return ee_frame.data.target_pos_w[..., 0, :]


def ee_env_pos(env: ManagerBasedEnv, asset_name: str) -> torch.Tensor:
    ee_frame_name = f"ee_frame_{asset_name}"
    ee_frame: FrameTransformer = env.scene[ee_frame_name]
    ee_frame_pos = ee_frame.data.target_pos_w[:, 0, :] - env.scene.env_origins[:, 0:3]
    return ee_frame_pos


def ee_euler_xyz(env: ManagerBasedEnv, asset_name: str) -> torch.Tensor:
    ee_frame_name = f"ee_frame_{asset_name}"
    ee_frame: FrameTransformer = env.scene[ee_frame_name]
    ee_quat = ee_frame.data.target_quat_w[..., 0, :]
    roll, pitch, yaw = euler_xyz_from_quat(ee_quat)
    return torch.stack([roll, pitch, yaw], dim=-1)


def object_euler_xyz(env: ManagerBasedEnv, asset_name: str) -> torch.Tensor:
    obj_name = f"{asset_name}"
    obj = env.scene[obj_name]
    roll, pitch, yaw = euler_xyz_from_quat(obj.data.root_quat_w)
    return torch.stack([roll, pitch, yaw], dim=-1)
