# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import Dict
from isaaclab.utils import configclass

from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.managers.action_manager import ActionTermCfg
from isaaclab.managers.event_manager import EventTermCfg
from isaaclab.sensors import FrameTransformerCfg
@configclass
class MatterixArticulationCfg(ArticulationCfg):
    """Configuration parameters for an articulation."""

    action_terms : Dict[str, ActionTermCfg] = {}

    event_terms : Dict[str, EventTermCfg] = {}

    semantic_tags: list[tuple] = []

    sensors : Dict[str, FrameTransformerCfg] = {}