# Copyright (c) 2022-2026, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Data structures for state machine scene state."""

from __future__ import annotations

import torch
from dataclasses import dataclass


@dataclass
class Pose:
    """Pose in world frame.

    Position in meters, orientation as quaternion (x, y, z, w).
    All tensors are batched across environments: (num_envs, ...).
    """

    position: torch.Tensor
    """Position in world frame. Shape: (num_envs, 3)"""

    orientation: torch.Tensor
    """Orientation as quaternion (x, y, z, w). Shape: (num_envs, 4)"""


@dataclass
class ArticulationData:
    """State of an articulation (robot, articulated object, etc.).

    All tensors are batched across environments with shape (num_envs, ...).
    This includes robots (arms + grippers) and articulated objects (doors, drawers).
    """

    root_pos_w: torch.Tensor
    """Root position in world frame. Shape: (num_envs, 3)"""

    root_quat_w: torch.Tensor
    """Root orientation (quaternion) in world frame. Shape: (num_envs, 4)"""

    joint_pos: torch.Tensor
    """Joint positions. Shape: (num_envs, num_joints)"""

    joint_vel: torch.Tensor
    """Joint velocities. Shape: (num_envs, num_joints)"""

    ee_pos_w: torch.Tensor | None = None
    """End-effector position in world frame (for robots). Shape: (num_envs, 3)"""

    ee_quat_w: torch.Tensor | None = None
    """End-effector orientation (quaternion) in world frame (for robots). Shape: (num_envs, 4)"""

    gripper_pos: torch.Tensor | None = None
    """Gripper opening/position (for robots). Shape: (num_envs, 1)"""


@dataclass
class RigidObjectData:
    """State of a rigid object (beaker, flask, etc.).

    All tensors are batched across environments with shape (num_envs, ...).
    """

    pos_w: torch.Tensor
    """Position in world frame. Shape: (num_envs, 3)"""

    quat_w: torch.Tensor
    """Orientation (quaternion) in world frame. Shape: (num_envs, 4)"""

    lin_vel_w: torch.Tensor | None = None
    """Linear velocity in world frame. Shape: (num_envs, 3)"""

    ang_vel_w: torch.Tensor | None = None
    """Angular velocity in world frame. Shape: (num_envs, 3)"""

    frames: dict[str, Pose] | None = None
    """Manipulation frames in world frame (e.g., 'grasp', 'pre_grasp', 'post_grasp').

    Note: Frames are defined as offsets in the object's body frame in asset configs,
    but are transformed to world frame by Isaac Lab's FrameTransformer before being
    stored in SceneData.
    """

    is_in_contact: torch.Tensor | None = None
    """Physics-derived contact state, shape (num_envs,), or None if not populated.

    Sourced from an ``IsInContactPhysicsCfg``/``IsInContactManual`` semantic attached to
    this object and exposed via an ``ObsTerm`` calling
    ``matterix.envs.mdp.observations.object_is_in_contact`` (see
    ``managers/semantics/primitive_semantics/is_in_contact.py``). None means the task's
    observation/semantics config never populated this field for this object -- actions
    that need it (e.g. ``VerifyContactCfg``) should raise a clear error rather than
    silently treating it as "not in contact".
    """


@dataclass
class SceneData:
    """Complete scene state container for state machine.

    This data structure is populated from environment observations and provides
    structured access to scene state for actions. It follows Isaac Lab naming
    conventions and organization.

    The structure is auto-created on the first observation and updated each step.
    """

    articulations: dict[str, ArticulationData]
    """Dictionary of articulation states keyed by name.
    Includes both robots and articulated objects (doors, drawers, etc.).
    Example: {'robot': ArticulationData, 'door': ArticulationData}
    """

    rigid_objects: dict[str, RigidObjectData]
    """Dictionary of rigid object states keyed by name.
    Example: {'beaker': RigidObjectData, 'flask': RigidObjectData}
    """

    # Future extensions (add as needed):
    # particle_systems: dict[str, ParticleSystemData] | None = None
    # sensors: dict[str, SensorData] | None = None
    # semantics: SemanticsData | None = None
