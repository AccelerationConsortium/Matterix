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
    VerifyContactCfg,
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
        verify_grasp: If True, append a VerifyContactCfg after the post-grasp lift that
            requires `object`'s physics-derived is_in_contact state to be True (i.e. the
            object actually moved with the gripper, not just that the gripper closed for
            long enough). Requires `object`'s scene config to expose is_in_contact via an
            IsInContactPhysicsCfg semantic + ObsTerm -- see VerifyContactCfg's docstring.
            Default: False (preserves prior pose/duration-only behavior).

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
    verify_grasp: bool = False

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

        if self.verify_grasp:
            # Checked AFTER the lift, not right after CloseGripper: fingers can register
            # contact against an object still resting on the table without actually
            # holding it. Requiring contact to persist through the lift is the real test.
            self.sub_actions.append(
                VerifyContactCfg(
                    object=self.object,
                    expected_contact=True,
                )
            )
