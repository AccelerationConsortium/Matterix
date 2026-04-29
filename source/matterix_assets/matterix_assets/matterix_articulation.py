# Copyright (c) 2022-2026, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for articulated objects."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matterix.managers.semantics.semantics_cfg import SemanticCfg
    from matterix.managers.semantics.semantic_presets import SemanticPreset

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

    mass: float | None = None  # Total mass in kg (for semantic simulations)

    action_terms: dict[str, ActionTermCfg] = {}

    event_terms: dict[str, EventTermCfg] = {}

    semantic_tags: list[tuple] = []  # Static metadata for classification (e.g., [("class", "robot")])
    semantics: "list[SemanticCfg] | SemanticPreset" = []  # Semantic behaviors; accepts a preset or a list of cfgs

    sensors: dict[str, FrameTransformerCfg] = {}

    def __post_init__(self):
        if hasattr(super(), "__post_init__"):
            super().__post_init__()

        # Normalize semantics: a SemanticPreset (or a list mixing presets and raw cfgs)
        # must be flattened to a plain list of SemanticsCfg instances.
        from matterix.managers.semantics.semantic_presets import SemanticPreset  # noqa: F811

        if isinstance(self.semantics, SemanticPreset):
            self.semantics = self.semantics.to_list()
        elif isinstance(self.semantics, list):
            flattened = []
            for item in self.semantics:
                if isinstance(item, SemanticPreset):
                    flattened.extend(item.to_list())
                else:
                    flattened.append(item)
            self.semantics = flattened

        if self.pos is not None:
            self.init_state.pos = self.pos
        if self.rot is not None:
            self.init_state.rot = self.rot
