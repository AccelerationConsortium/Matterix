# Copyright (c) 2022-2026, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PickObject compositional action - pick an object using frame-based manipulation."""

from __future__ import annotations

from dataclasses import MISSING

from .._compat import configclass
from ..compositional_action import CompositionalActionCfg
from ..primitive_actions import (
    CloseGripperCfg,
    MoveRelativeCfg,
    MoveToFrameCfg,
    OpenGripperCfg,
)
from ..robot_action_spaces import ActionSpaceInfo


@configclass
class PickObjectCfg(CompositionalActionCfg):
    """Configuration for PickObject compositional action.

    Performs: MoveToFrame(pre_grasp) -> OpenGripper -> MoveToFrame(grasp) ->
              CloseGripper -> MoveRelative(post_grasp)

    Attributes:
        agent_assets: Name(s) of articulated asset(s) acting as agents (e.g., robot manipulators). REQUIRED.
        object: Name of the object to pick. REQUIRED.
        post_grasp_offset: Offset for post-grasp lift (x, y, z). Defaults to (0, 0, 0.1).
        action_space_info: Optional action space metadata.

    Note:
        num_envs, device, and dt are NOT in compositional configs - they're set by StateMachine
        on the primitive sub-actions automatically.
    """

    # Required fields
    agent_assets: str | list[str] = MISSING
    object: str = MISSING

    # Optional fields with defaults
    post_grasp_offset: tuple[float, float, float] = (0.0, 0.0, 0.1)
    action_space_info: ActionSpaceInfo | None = None

    def __post_init__(self):
        """Generate default sub_actions for pick sequence after initialization."""
        super().__post_init__()

        # Generate the standard 5-action pick sequence
        # Uses default thresholds from MoveToPoseCfg (0.01m position, 0.02rad orientation)
        self.sub_actions = [
            OpenGripperCfg(
                agent_assets=self.agent_assets,
                action_space_info=self.action_space_info,
            ),
            MoveToFrameCfg(
                object=self.object,
                frame="pre_grasp",
                agent_assets=self.agent_assets,
                action_space_info=self.action_space_info,
            ),
            MoveToFrameCfg(
                object=self.object,
                frame="grasp",
                agent_assets=self.agent_assets,
                action_space_info=self.action_space_info,
            ),
            CloseGripperCfg(
                agent_assets=self.agent_assets,
                action_space_info=self.action_space_info,
            ),
            MoveRelativeCfg(
                agent_assets=self.agent_assets,
                position_offset=self.post_grasp_offset,
                orientation_offset=None,
                action_space_info=self.action_space_info,
            ),
        ]
