# Copyright (c) 2022-2026, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""State machine for sequential action orchestration.

This module provides a hierarchical action system with primitive and compositional actions:

Primitive Actions (in primitive_actions/):
    - Move: Move to target world positions
    - MoveToFrame: Move to an object's named frame
    - MoveRelative: Move relative to current position
    - GripperAction, OpenGripper, CloseGripper: Gripper control

Compositional Actions (in compositional_actions/):
    - CompositionalAction: Base class for action sequences
    - PickObject: Pick an object using frame-based manipulation

Core Components:
    - ActionBase: Base class for all actions
    - StateMachine: Orchestrates action sequences across parallel environments
    - SceneData: Container for all scene state (replaces WorkflowEnv interface)
"""

# Import base classes
from .action_base import ActionBase, ActionBaseCfg
from .action_constants import FRANKA_ORIENTATION_DOWN_QUAT

# Import compositional actions
from .compositional_actions import (
    CompositionalAction,
    CompositionalActionCfg,
    PickObjectCfg,
)

# Import math utilities
from .math_utils import quat_mul, quat_rotate
from .primitive_action import PrimitiveAction, PrimitiveActionCfg

# Import primitive actions
from .primitive_actions import (
    CloseGripper,
    CloseGripperCfg,
    GripperAction,
    GripperActionCfg,
    MoveRelative,
    MoveRelativeCfg,
    MoveToFrame,
    MoveToFrameCfg,
    OpenGripper,
    OpenGripperCfg,
)
from .scene_data import ArticulationData, Pose, RigidObjectData, SceneData
from .state_machine import StateMachine

__all__ = [
    # Base classes
    "ActionBase",
    "ActionBaseCfg",
    "PrimitiveAction",
    "PrimitiveActionCfg",
    "StateMachine",
    # Data structures
    "Pose",
    "SceneData",
    "ArticulationData",
    "RigidObjectData",
    # Math utilities
    "quat_mul",
    "quat_rotate",
    # Constants
    "FRANKA_ORIENTATION_DOWN_QUAT",
    # Primitive actions
    "MoveToFrame",
    "MoveToFrameCfg",
    "MoveRelative",
    "MoveRelativeCfg",
    "GripperAction",
    "GripperActionCfg",
    "OpenGripper",
    "OpenGripperCfg",
    "CloseGripper",
    "CloseGripperCfg",
    # Compositional actions
    "CompositionalAction",
    "CompositionalActionCfg",
    "PickObjectCfg",
]
