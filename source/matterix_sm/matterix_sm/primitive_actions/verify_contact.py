# Copyright (c) 2022-2026, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""VerifyContact action - checks a rigid object's physics-derived contact state."""

from __future__ import annotations

import torch
from dataclasses import MISSING, field
from typing import ClassVar

from .._compat import configclass
from ..primitive_action import PrimitiveAction, PrimitiveActionCfg
from ..scene_data import SceneData


@configclass
class VerifyContactCfg(PrimitiveActionCfg):
    """Configuration for VerifyContact action.

    Checks a rigid object's ``is_in_contact`` state (populated in SceneData from an
    ``IsInContactPhysicsCfg``/``IsInContactManual`` semantic, exposed via an ObsTerm
    calling ``mdp.observations.object_is_in_contact``) against an expected value. Does
    not move the robot -- the StateMachine preserves whatever action_dict values were
    set by the previous action (same mechanism WaitCfg uses).

    Use ``expected_contact=True`` after closing the gripper on an object to verify it
    was actually grasped (not just that the gripper closed for long enough). Use
    ``expected_contact=False`` after opening the gripper to verify the object actually
    released (not stuck to a finger via friction/adhesion) instead of assuming success
    from gripper-open duration alone.

    Attributes:
        agent_assets: Empty by default (no robots controlled). Leave as default.
        object: Name of the rigid object whose is_in_contact state to check. REQUIRED.
        expected_contact: True to require contact (grasp verification), False to require
            no contact (release verification). Default: True.
        settling_time: Time (in seconds) the contact state must hold continuously before
            success. Filters single-frame contact-sensor flicker. Default: 0.1s.

    Note:
        Requires the named object's scene config to attach a contact semantic (e.g.
        ``IsInContactPhysicsCfg`` with ``activate_contact_sensors=True`` on the relevant
        assets) AND expose it via an ObsTerm using
        ``mdp.observations.object_is_in_contact``. If that plumbing is missing, the
        action raises ValueError at check time rather than silently timing out.

    Example::

        PickObjectCfg(agent_assets="robot", object="beaker", verify_grasp=True)
        # is equivalent to appending, after the pick sequence's lift:
        VerifyContactCfg(object="beaker", expected_contact=True)
    """

    agent_assets: str | list[str] = field(default_factory=list)
    object: str = MISSING
    expected_contact: bool = True
    settling_time: float = 0.1


class VerifyContact(PrimitiveAction):
    """Check a rigid object's physics-derived contact state against an expected value."""

    cfg_type: ClassVar[type] = VerifyContactCfg

    def __init__(
        self,
        object: str,
        expected_contact: bool = True,
        settling_time: float = 0.1,
        timeout: float | None = None,
        agent_assets: str | list[str] | None = None,
        action_space_info=None,
        semantics=None,
    ):
        """
        Args:
            object: Name of the rigid object whose is_in_contact state to check.
            expected_contact: True to require contact, False to require no contact.
            settling_time: Time (in seconds) the state must hold before success.
            timeout: Max time (in seconds) before timeout. Uses PrimitiveActionCfg's
                default (TIMEOUT_DEFAULT) if None.
            agent_assets: Unused (always empty) -- kept for from_cfg() symmetry.
            action_space_info: Unused (this action never moves the robot).
            semantics: Optional semantic actions to emit on success.
        """
        from ..action_constants import TIMEOUT_DEFAULT

        super().__init__(
            agent_assets or [], timeout if timeout is not None else TIMEOUT_DEFAULT, action_space_info, semantics
        )
        self.object = object
        self.expected_contact = expected_contact
        self.settling_time = settling_time

        # Initialized in set_execution_params()
        self.time_in_state = None

    def set_execution_params(self, num_envs: int, device: str | torch.device, dt: float) -> None:
        """Set execution parameters and initialize the settling timer."""
        super().set_execution_params(num_envs, device, dt)
        self.time_in_state = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

    def _compute_action_impl(self, scene_data: SceneData, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return dummy tensors (robot holds last commanded state).

        Since agent_assets=[], StateMachine never uses these values -- the robot
        continues executing its last commanded action unchanged, same as WaitCfg.
        """
        action_tensor = torch.zeros(self.num_envs, 1, device=self.device)
        action_mask = torch.zeros(1, dtype=torch.bool, device=self.device)
        return action_tensor, action_mask

    def _check_completion_impl(self, scene_data: SceneData, env_ids: torch.Tensor) -> None:
        """Check if the object's contact state matches expected_contact, settled.

        Args:
            scene_data: Complete scene state container.
            env_ids: Indices of active environments.

        Raises:
            ValueError: If the object is missing from scene_data, or has no
                is_in_contact data -- both are scene-authoring gaps this action cannot
                recover from, so it fails loudly instead of silently timing out.
        """
        if self.object not in scene_data.rigid_objects:
            raise ValueError(
                f"VerifyContact: object '{self.object}' not found in scene_data.rigid_objects. "
                f"Available: {list(scene_data.rigid_objects.keys())}"
            )

        is_in_contact = scene_data.rigid_objects[self.object].is_in_contact
        if is_in_contact is None:
            raise ValueError(
                f"VerifyContact: '{self.object}' has no is_in_contact data. Attach an "
                "IsInContactPhysicsCfg (or IsInContactManual) semantic to it -- with "
                "activate_contact_sensors=True on the relevant assets for the physics "
                "variant -- and expose it via an ObsTerm using "
                "mdp.observations.object_is_in_contact so it reaches the state machine."
            )

        in_expected_state = is_in_contact.to(self.device) == self.expected_contact

        # Settling timer: accumulate while in the expected state, reset otherwise --
        # same pattern as MoveToPose's position/orientation settling check.
        self.time_in_state[env_ids] = torch.where(
            in_expected_state[env_ids],
            self.time_in_state[env_ids] + self.dt,
            torch.zeros_like(self.time_in_state[env_ids]),
        )

        self._env_success_mask[env_ids] = self.time_in_state[env_ids] >= self.settling_time
        self._env_failure_mask[env_ids] = False

    def _reset_impl(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset the settling timer when environments are reset."""
        if env_ids is None:
            self.time_in_state.zero_()
        else:
            self.time_in_state[env_ids] = 0.0

    @classmethod
    def from_cfg(cls, cfg: VerifyContactCfg) -> VerifyContact:
        """Create VerifyContact from configuration."""
        return cls(
            object=cfg.object,
            expected_contact=cfg.expected_contact,
            settling_time=cfg.settling_time,
            timeout=cfg.timeout,
            agent_assets=cfg.agent_assets,
            action_space_info=cfg.action_space_info,
            semantics=cfg.semantics,
        )
