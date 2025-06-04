# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from isaaclab.utils import configclass

from isaaclab.assets.articulation import ArticulationCfg

@configclass
class MatterixArticulationCfg(ArticulationCfg):
    """Configuration parameters for an articulation."""

    action_terms = {}

    event_terms = {}

    semantic_tags: list[tuple] = []