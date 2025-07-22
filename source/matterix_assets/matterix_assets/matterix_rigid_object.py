# Copyright (c) 2022-2025, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import Dict, List

from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.managers.event_manager import EventTermCfg


@configclass
class MatterixRigidObject(RigidObjectCfg):
    """Configuration parameters for an articulation."""

    event_terms: dict[str, EventTermCfg] = {}

    frames: dict[str, list[int]] = {} # TODEL
    sensors: dict[str, FrameTransformerCfg] = {}

    semantic_tags: list[tuple] = []
