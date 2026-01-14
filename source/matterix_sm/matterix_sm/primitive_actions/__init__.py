# Copyright (c) 2022-2026, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Primitive actions for the state machine.

This module provides the basic building blocks for robot manipulation:
- PrimitiveAction: Base class for all primitive actions
- MoveToPose: Move to target world pose (position + orientation)
- MoveToFrame: Move to an object's named frame
- MoveRelative: Move relative to current position
- GripperAction, OpenGripper, CloseGripper: Gripper control

Each action is paired with its configuration class (e.g., MoveToPose + MoveToPoseCfg).
"""

from ..primitive_action import PrimitiveAction
from .gripper import (
    CloseGripper,
    CloseGripperCfg,
    GripperAction,
    GripperActionCfg,
    OpenGripper,
    OpenGripperCfg,
)
from .move_relative import MoveRelative, MoveRelativeCfg
from .move_to_frame import MoveToFrame, MoveToFrameCfg
from .move_to_pose import MoveToPose, MoveToPoseCfg

__all__ = [
    # Base class
    "PrimitiveAction",
    # Move actions
    "MoveToPose",
    "MoveToPoseCfg",
    "MoveToFrame",
    "MoveToFrameCfg",
    "MoveRelative",
    "MoveRelativeCfg",
    # Gripper actions
    "GripperAction",
    "GripperActionCfg",
    "OpenGripper",
    "OpenGripperCfg",
    "CloseGripper",
    "CloseGripperCfg",
]
