# Copyright (c) 2022-2026, The Matterix Project Developers.
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


def root_world_pos(env: ManagerBasedEnv, asset_name: str) -> torch.Tensor:
    """Root position in world frame."""
    asset = env.scene[asset_name]
    return asset.data.root_pos_w


def root_world_quat(env: ManagerBasedEnv, asset_name: str) -> torch.Tensor:
    """Root orientation (quaternion) in world frame."""
    asset = env.scene[asset_name]
    return asset.data.root_quat_w


def joint_pos(env: ManagerBasedEnv, asset_name: str) -> torch.Tensor:
    """Joint positions."""
    asset = env.scene[asset_name]
    return asset.data.joint_pos


def joint_vel(env: ManagerBasedEnv, asset_name: str) -> torch.Tensor:
    """Joint velocities."""
    asset = env.scene[asset_name]
    return asset.data.joint_vel


def ee_world_quat(env: ManagerBasedEnv, asset_name: str) -> torch.Tensor:
    """End-effector orientation (quaternion) in world frame."""
    ee_frame_name = f"ee_frame_{asset_name}"
    ee_frame: FrameTransformer = env.scene[ee_frame_name]
    return ee_frame.data.target_quat_w[..., 0, :]


def gripper_pos(env: ManagerBasedEnv, asset_name: str) -> torch.Tensor:
    """Obtain the versatile gripper position of both Gripper and Suction Cup.

    Follows Isaac Lab pattern: checks for gripper_joint_names in env config,
    returns joint positions for those joints.
    """
    asset = env.scene[asset_name]

    # Handle surface grippers (suction cups) if present
    if hasattr(env.scene, "surface_grippers") and len(env.scene.surface_grippers) > 0:
        gripper_states = []
        for gripper_name, surface_gripper in env.scene.surface_grippers.items():
            gripper_states.append(surface_gripper.state.view(-1, 1))

        if len(gripper_states) == 1:
            return gripper_states[0]
        else:
            return torch.cat(gripper_states, dim=1)

    # Handle articulated grippers
    else:
        if hasattr(env.cfg, "gripper_joint_names"):
            gripper_joint_ids, _ = asset.find_joints(env.cfg.gripper_joint_names)
            assert len(gripper_joint_ids) == 2, "Observation gripper_pos only supports parallel gripper for now"
            finger_joint_1 = asset.data.joint_pos[:, gripper_joint_ids[0]].clone().unsqueeze(1)
            finger_joint_2 = -1 * asset.data.joint_pos[:, gripper_joint_ids[1]].clone().unsqueeze(1)
            return torch.cat((finger_joint_1, finger_joint_2), dim=1)
        else:
            raise NotImplementedError("[Error] Cannot find gripper_joint_names in the environment config")


##
# Rigid Object Observations
##


def object_world_pos(env: ManagerBasedEnv, asset_name: str) -> torch.Tensor:
    """Object position in world frame."""
    obj = env.scene[asset_name]
    return obj.data.root_pos_w


def object_world_quat(env: ManagerBasedEnv, asset_name: str) -> torch.Tensor:
    """Object orientation (quaternion) in world frame."""
    obj = env.scene[asset_name]
    return obj.data.root_quat_w


def object_lin_vel(env: ManagerBasedEnv, asset_name: str) -> torch.Tensor:
    """Object linear velocity in world frame."""
    obj = env.scene[asset_name]
    return obj.data.root_lin_vel_w


def object_ang_vel(env: ManagerBasedEnv, asset_name: str) -> torch.Tensor:
    """Object angular velocity in world frame."""
    obj = env.scene[asset_name]
    return obj.data.root_ang_vel_w


def frame_world_pos(env: ManagerBasedEnv, asset_name: str, frame_name: str) -> torch.Tensor:
    """Get frame position in world frame for a specific asset and frame.

    Args:
        env: The environment instance.
        asset_name: Name of the asset (e.g., "beaker", "robot").
        frame_name: Name of the frame (e.g., "grasp", "pre_grasp", "post_grasp").

    Returns:
        Position tensor of shape (num_envs, 3) in world frame.

    Note:
        Frame transformers in Isaac Lab are stored with pattern: {frame_name}_{asset_name}
    """
    frame_key = f"{frame_name}_{asset_name}"
    frame_transformer: FrameTransformer = env.scene[frame_key]
    return frame_transformer.data.target_pos_w[..., 0, :]


def frame_world_quat(env: ManagerBasedEnv, asset_name: str, frame_name: str) -> torch.Tensor:
    """Get frame orientation (quaternion) in world frame for a specific asset and frame.

    Args:
        env: The environment instance.
        asset_name: Name of the asset (e.g., "beaker", "robot").
        frame_name: Name of the frame (e.g., "grasp", "pre_grasp", "post_grasp").

    Returns:
        Quaternion tensor of shape (num_envs, 4) in world frame as (w, x, y, z).

    Note:
        Frame transformers in Isaac Lab are stored with pattern: {frame_name}_{asset_name}
    """
    frame_key = f"{frame_name}_{asset_name}"
    frame_transformer: FrameTransformer = env.scene[frame_key]
    return frame_transformer.data.target_quat_w[..., 0, :]


def frame_world_pose(env: ManagerBasedEnv, asset_name: str, frame_name: str) -> torch.Tensor:
    """Get frame pose (position + quaternion) in world frame for a specific asset and frame.

    Args:
        env: The environment instance.
        asset_name: Name of the asset (e.g., "beaker", "robot").
        frame_name: Name of the frame (e.g., "grasp", "pre_grasp", "post_grasp").

    Returns:
        Pose tensor of shape (num_envs, 7) as [x, y, z, qw, qx, qy, qz] in world frame.

    Note:
        Frame transformers in Isaac Lab are stored with pattern: {frame_name}_{asset_name}
    """
    frame_key = f"{frame_name}_{asset_name}"
    frame_transformer: FrameTransformer = env.scene[frame_key]
    pos = frame_transformer.data.target_pos_w[..., 0, :]  # (num_envs, 3)
    quat = frame_transformer.data.target_quat_w[..., 0, :]  # (num_envs, 4)
    return torch.cat([pos, quat], dim=-1)  # (num_envs, 7)


def object_frames(env: ManagerBasedEnv, asset_name: str) -> dict:
    """Extract frame transformations for an object.

    Returns a dictionary of frame names to their world poses (pos + quat).
    Example: {"grasp": tensor([x, y, z, qw, qx, qy, qz]), ...}
    """
    # Get the asset config to find which frames are defined
    asset = env.scene[asset_name]

    # Try to get frame names from the asset config
    # Frames are stored as sensors with naming pattern: {frame_name}_{asset_name}
    frames_dict = {}

    # Check if asset has sensors attribute with frame transformers
    if hasattr(asset, "cfg") and hasattr(asset.cfg, "sensors"):
        for frame_name in asset.cfg.sensors.keys():
            # Build the scene key for this frame (Isaac Lab uses frame_name BEFORE asset_name)
            frame_key = f"{frame_name}_{asset_name}"
            try:
                frame_transformer: FrameTransformer = env.scene[frame_key]
                # Get position and quaternion in world frame
                pos = frame_transformer.data.target_pos_w[:, 0, :]  # (num_envs, 3)
                quat = frame_transformer.data.target_quat_w[:, 0, :]  # (num_envs, 4)
                # Concatenate into 7D tensor
                frames_dict[frame_name] = torch.cat([pos, quat], dim=-1)  # (num_envs, 7)
            except KeyError:
                # Frame transformer not found in scene, skip it
                pass

    return frames_dict


def object_all_frames(env: ManagerBasedEnv, asset_name: str) -> dict[str, torch.Tensor]:
    """Get all manipulation frames for an object in world coordinates.

    Returns frame transformations continuously computed by FrameTransformers.
    Each frame is returned as a 7D tensor: [pos(3), quat(4)] in world frame.

    Args:
        env: The environment instance.
        asset_name: Name of the asset (e.g., "beaker", "flask").

    Returns:
        Dictionary mapping frame names to 7D pose tensors (num_envs, 7).
        Example: {"pre_grasp": tensor([x, y, z, qw, qx, qy, qz]), ...}
    """
    asset = env.scene[asset_name]

    # Check if asset has frame transformers defined
    if not (hasattr(asset, "cfg") and hasattr(asset.cfg, "sensors")):
        return {}

    sensor_names = list(asset.cfg.sensors.keys())
    frames_dict = {}

    for frame_name in sensor_names:
        # Frame transformers are stored with pattern: {frame_name}_{asset_name}
        # NOTE: Isaac Lab uses frame_name BEFORE asset_name
        frame_key = f"{frame_name}_{asset_name}"
        try:
            frame_transformer: FrameTransformer = env.scene[frame_key]
            # Get position and quaternion in world frame
            pos = frame_transformer.data.target_pos_w[:, 0, :]  # (num_envs, 3)
            quat = frame_transformer.data.target_quat_w[:, 0, :]  # (num_envs, 4)
            # Concatenate into 7D tensor [x, y, z, qw, qx, qy, qz]
            frames_dict[frame_name] = torch.cat([pos, quat], dim=-1)  # (num_envs, 7)
        except KeyError:
            # Frame transformer not found in scene, skip it
            pass

    return frames_dict
