# Copyright (c) 2022-2026, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PlaceObject compositional action - place a held object on top of a target object."""

from __future__ import annotations

from dataclasses import MISSING

from .._compat import configclass
from ..compositional_action import CompositionalActionCfg
from ..primitive_actions import (
    MoveRelativeCfg,
    MoveToFrameCfg,
    OpenGripperCfg,
    VerifyContactCfg,
)
from ..robot_action_spaces import ActionSpaceInfo


@configclass
class PlaceObjectCfg(CompositionalActionCfg):
    """Configuration for PlaceObject compositional action.

    Performs: MoveToFrame(pre_place on target) -> MoveToFrame(place on target) ->
              OpenGripper -> MoveRelative(post_place)

    Attributes:
        agent_assets: Name(s) of articulated asset(s) acting as agents (e.g., robot manipulators). REQUIRED.
        target: Name of the target object to place on. REQUIRED.
        post_place_offset: Offset for post-place retreat (x, y, z). Defaults to (0, 0, 0.1).
        action_space_info: Optional action space metadata.
        held_object: Name of the object being placed (i.e. whatever a prior PickObjectCfg
            in the same workflow picked up). Only needed when verify_release=True --
            PlaceObjectCfg otherwise has no way to know which object's contact state to
            check, since it only ever references `target`, not what the gripper is
            holding. Default: None.
        verify_release: If True, append a VerifyContactCfg after the post-place retreat
            that requires `held_object`'s physics-derived is_in_contact state to be
            False (i.e. it actually let go, not stuck to a finger via friction/adhesion).
            Requires `held_object` to be set, and its scene config to expose
            is_in_contact via an IsInContactPhysicsCfg semantic + ObsTerm -- see
            VerifyContactCfg's docstring. Default: False (preserves prior
            pose/duration-only behavior).

    Note:
        num_envs, device, and dt are NOT in compositional configs - they're set by StateMachine
        on the primitive sub-actions automatically.
    """

    # Required fields
    agent_assets: str | list[str] = MISSING
    target: str = MISSING

    # Optional fields with defaults
    post_place_offset: tuple[float, float, float] = (0.0, 0.0, 0.1)
    action_space_info: ActionSpaceInfo | None = None
    held_object: str | None = None
    verify_release: bool = False

    def __post_init__(self):
        """Generate default sub_actions for place sequence after initialization."""
        super().__post_init__()

        # Generate the standard 4-action place sequence.
        # use_frame_orientation=False: keep the robot's current orientation from grasping
        # rather than matching the target object's orientation (which may differ).
        self.sub_actions = [
            MoveToFrameCfg(
                object=self.target,
                frame="pre_place",
                agent_assets=self.agent_assets,
                use_frame_orientation=False,
                action_space_info=self.action_space_info,
            ),
            MoveToFrameCfg(
                object=self.target,
                frame="place",
                agent_assets=self.agent_assets,
                use_frame_orientation=False,
                action_space_info=self.action_space_info,
            ),
            OpenGripperCfg(
                agent_assets=self.agent_assets,
                action_space_info=self.action_space_info,
            ),
            MoveRelativeCfg(
                agent_assets=self.agent_assets,
                position_offset=self.post_place_offset,
                orientation_offset=None,
                action_space_info=self.action_space_info,
            ),
        ]

        if self.verify_release:
            if self.held_object is None:
                raise ValueError(
                    "PlaceObjectCfg.verify_release=True requires held_object to be set "
                    "to the name of the object being placed -- PlaceObjectCfg only ever "
                    "references `target`, so without held_object it has no way to know "
                    "which object's contact state to check on release."
                )
            self.sub_actions.append(
                VerifyContactCfg(
                    object=self.held_object,
                    expected_contact=False,
                )
            )
