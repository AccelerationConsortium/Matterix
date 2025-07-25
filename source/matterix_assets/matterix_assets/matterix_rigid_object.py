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
class MatterixRigidObjectCfg(RigidObjectCfg):
    """Configuration parameters for a rigid object."""
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
   
    prim_path = "{ENV_REGEX_NS}/RigidObjects"

    event_terms: dict[str, EventTermCfg] = {}

    frames: dict[str, list[int]] = {} # TODEL
    sensors: dict[str, FrameTransformerCfg] = {}

    semantic_tags: list[tuple] = []
    
    def __post_init__(self):
        if hasattr(super(), "__post_init__"):
            super().__post_init__()

        self.init_state.pos = self.pos
        self.init_state.rot = self.rot

