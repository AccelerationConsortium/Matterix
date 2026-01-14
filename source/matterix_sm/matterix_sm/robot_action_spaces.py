# Copyright (c) 2022-2026, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action space definitions for common robots.

This module defines action space metadata that describes which indices of the action tensor
control which robot degrees of freedom (position, orientation, gripper, etc.).

This metadata is used by primitive actions to build masks for partial action updates,
allowing actions like GripperAction to only modify the gripper dimension without
affecting position/orientation set by other actions.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ActionSpaceInfo:
    """Metadata about action space structure for a robot.

    Defines which indices of the action tensor control which robot DOFs.
    Used by actions to build masks for partial updates.

    Attributes:
        total_dim: Total action dimension (e.g., 8 for pos+ori+gripper)
        position_indices: Indices for EE position control (e.g., (0,1,2))
        orientation_indices: Indices for EE orientation control (e.g., (3,4,5,6))
        gripper_indices: Indices for gripper control (e.g., (7,))
        joint_indices: Indices for direct joint control (optional, for joint-level control)
        grasp_to_ee_offset: Transform from grasp frame to EE frame (pos, quat).
                           This is the robot-specific offset that accounts for gripper geometry.
                           Format: tuple of (position_offset, quaternion_offset) where:
                             - position_offset: (3,) tuple (x, y, z) in meters
                             - quaternion_offset: (4,) tuple (w, x, y, z)

    Example:
        >>> # Standard Franka IK action space: [pos(3), quat(4), gripper(1)]
        >>> info = ActionSpaceInfo(
        ...     total_dim=8,
        ...     position_indices=(0, 1, 2),
        ...     orientation_indices=(3, 4, 5, 6),
        ...     gripper_indices=(7,),
        ...     grasp_to_ee_offset=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),  # 10.34cm offset
        ... )
    """

    total_dim: int
    position_indices: tuple[int, ...] | None = None
    orientation_indices: tuple[int, ...] | None = None
    gripper_indices: tuple[int, ...] | None = None
    joint_indices: tuple[int, ...] | None = None
    grasp_to_ee_offset: tuple[tuple[float, float, float], tuple[float, float, float, float]] | None = None


# ==================================================================================
# Pre-defined action spaces for common robots
# ==================================================================================

FRANKA_IK_ACTION_SPACE = ActionSpaceInfo(
    total_dim=8,
    position_indices=(0, 1, 2),
    orientation_indices=(3, 4, 5, 6),
    gripper_indices=(7,),
    grasp_to_ee_offset=(
        (0.0, 0.0, 0.0),  # 10.34cm offset along z-axis (typical for Franka gripper)
        (0.0, -1.0, 0.0, 0.0),  # Identity rotation (no rotation offset)
    ),
)
"""Standard Franka Panda with IK control (8-dim: pos + ori + gripper).

The grasp_to_ee_offset represents the transform from the grasping point (center between fingers)
to the end-effector frame (wrist flange). For Franka, this is approximately 10.34cm along the z-axis.
"""


# Add more robots as needed:
# UR5_IK_ACTION_SPACE = ActionSpaceInfo(...)
# CUSTOM_ROBOT_ACTION_SPACE = ActionSpaceInfo(...)
