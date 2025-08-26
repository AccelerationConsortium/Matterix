# Copyright (c) 2022-2025, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.managers.action_manager import ActionTermCfg
from isaaclab.managers.event_manager import EventTermCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.utils import configclass


@configclass
class MatterixArticulationCfg(ArticulationCfg):
    """Configuration parameters for an articulation."""

    pos: tuple[float, float, float] = None
    rot: tuple[float, float, float, float] = None

    prim_path = "{ENV_REGEX_NS}/Articulations"

    action_terms: dict[str, ActionTermCfg] = {}

    event_terms: dict[str, EventTermCfg] = {}

    semantic_tags: list[tuple] = []

    sensors: dict[str, FrameTransformerCfg] = {}

    def __post_init__(self):
        if hasattr(super(), "__post_init__"):
            super().__post_init__()
        if self.pos is not None:
            self.init_state.pos = self.pos
        if self.rot is not None:
            self.init_state.rot = self.rot
