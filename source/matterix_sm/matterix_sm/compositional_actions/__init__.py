# Copyright (c) 2022-2026, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compositional actions for the state machine.

This module provides high-level actions composed of primitive actions:
- CompositionalAction: Base class for all compositional actions
- PickObject: Pick an object using frame-based manipulation
- PlaceObject: Place a held object on top of a target object

Each compositional action is paired with its configuration class
(e.g., PickObject + PickObjectCfg).
"""

from ..compositional_action import CompositionalAction, CompositionalActionCfg
from .pick_object import PickObjectCfg
from .place_object import PlaceObjectCfg

__all__ = [
    # Base class
    "CompositionalAction",
    "CompositionalActionCfg",
    # Specific compositional action configs
    "PickObjectCfg",
    "PlaceObjectCfg",
]
