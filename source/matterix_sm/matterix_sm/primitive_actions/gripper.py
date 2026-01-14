# Copyright (c) 2022-2026, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gripper actions - open and close gripper while holding pose."""

from __future__ import annotations

import torch
from dataclasses import MISSING
from typing import ClassVar

from .._compat import configclass
from ..primitive_action import PrimitiveAction, PrimitiveActionCfg
from ..robot_action_spaces import ActionSpaceInfo
from ..scene_data import SceneData


@configclass
class GripperActionCfg(PrimitiveActionCfg):
    """Configuration for gripper actions (open/close).

    Inherits base fields (assets, timeout) from PrimitiveActionCfg.

    Attributes:
        target_value: Gripper command value (1.0 for open, -1.0 for close). REQUIRED.
        duration: Time (in seconds) to hold the command. Default: 1.0 second.
    """

    target_value: float = MISSING
    duration: float = 2.0


@configclass
class OpenGripperCfg(GripperActionCfg):
    """Configuration for OpenGripper action."""

    target_value: float = 1.0


@configclass
class CloseGripperCfg(GripperActionCfg):
    """Configuration for CloseGripper action."""

    target_value: float = -1.0


class GripperAction(PrimitiveAction):
    """Hold current EE pose and apply a binary gripper command for a fixed duration."""

    cfg_type: ClassVar[type] = GripperActionCfg

    def __init__(
        self,
        agent_assets: str | list[str],
        target_value: float,
        duration: float,
        timeout: float,
        action_space_info: ActionSpaceInfo | None = None,
    ):
        """
        Args:
            agent_assets: Name(s) of articulated asset(s) acting as agents.
            target_value: Gripper command value.
            duration: Time (in seconds) to maintain the command.
            timeout: Max time (in seconds) before timeout.
            action_space_info: Optional action space metadata for mask creation.
        """
        super().__init__(agent_assets, timeout, action_space_info)
        self.target_value = target_value
        self.duration = duration

        # Validate action_space_info at init (fail-fast)
        if self.action_space_info is None:
            raise ValueError(
                "GripperAction requires action_space_info to determine gripper indices. "
                "Pass action_space_info parameter when creating the action."
            )

        # Cache gripper indices for fast access
        self._gripper_indices = (
            list(self.action_space_info.gripper_indices) if self.action_space_info.gripper_indices is not None else []
        )

        # These will be initialized in set_execution_params()
        self._action_dim_mask = None
        self._action_tensor = None

    def set_execution_params(self, num_envs: int, device: str | torch.device, dt: float) -> None:
        """Set execution parameters and initialize gripper-specific tensors."""
        super().set_execution_params(num_envs, device, dt)

        # Cache the action dimension mask (computed once, reused every step)
        self._action_dim_mask = self._create_action_mask("gripper")

        # Pre-allocate action tensor (will be zeroed and reused each step)
        self._action_tensor = torch.zeros((self.num_envs, self.action_space_info.total_dim), device=self.device)

    def _compute_action_impl(self, scene_data: SceneData, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute gripper action for controlled asset.

        Args:
            scene_data: Complete scene state container.
            env_ids: Indices of active environments.

        Returns:
            (action_tensor, action_dim_mask):
                - action_tensor: Shape (num_envs, action_dim) - action values for all envs
                - action_dim_mask: Shape (action_dim,) - which dimensions this action controls
        """
        # action_space_info is validated in __init__, guaranteed to be non-None
        assert self.action_space_info is not None, "action_space_info must be provided"

        # Zero out the action tensor for active environments
        # The StateMachine will use mask-based accumulation to preserve
        # position/orientation from the previous action (last-write-wins)
        self._action_tensor[env_ids] = 0.0

        # Set only gripper dimension(s) to target value - only for active envs
        # Position/orientation will be preserved from action_dict by the mask
        for idx in self._gripper_indices:
            self._action_tensor[env_ids, idx] = self.target_value

        return self._action_tensor, self._action_dim_mask

    def _check_completion_impl(self, scene_data: SceneData, env_ids: torch.Tensor) -> None:
        """Check if gripper action duration elapsed.

        Args:
            scene_data: Complete scene state container (unused for gripper).
            env_ids: Indices of active environments.
        """
        # Success when duration (in seconds) elapsed
        self._env_success_mask[env_ids] = self.time_elapsed[env_ids] >= self.duration
        # No custom failure modes
        self._env_failure_mask[env_ids] = False

    @classmethod
    def from_cfg(cls, cfg: GripperActionCfg):
        """Create GripperAction from configuration."""
        return cls(
            agent_assets=cfg.agent_assets,
            target_value=cfg.target_value,
            duration=cfg.duration,
            timeout=cfg.timeout,
            action_space_info=cfg.action_space_info,
        )


class OpenGripper(GripperAction):
    """Open the gripper while holding the current pose."""

    cfg_type: ClassVar[type] = OpenGripperCfg

    def __init__(
        self,
        agent_assets: str | list[str],
        duration: float,
        timeout: float,
        target_value: float = 1.0,
        action_space_info: ActionSpaceInfo | None = None,
    ):
        super().__init__(
            agent_assets=agent_assets,
            target_value=target_value,
            duration=duration,
            timeout=timeout,
            action_space_info=action_space_info,
        )

    @classmethod
    def from_cfg(cls, cfg: OpenGripperCfg):
        """Create OpenGripper from configuration.

        Note: target_value is hardcoded to 1.0 in __init__, so we don't pass it from cfg.
        """
        return cls(
            agent_assets=cfg.agent_assets,
            duration=cfg.duration,
            timeout=cfg.timeout,
            target_value=cfg.target_value,
            action_space_info=cfg.action_space_info,
        )


class CloseGripper(GripperAction):
    """Close the gripper while holding the current pose."""

    cfg_type: ClassVar[type] = CloseGripperCfg

    def __init__(
        self,
        agent_assets: str | list[str],
        duration: float,
        timeout: float,
        target_value: float = -1.0,
        action_space_info: ActionSpaceInfo | None = None,
    ):
        super().__init__(
            agent_assets=agent_assets,
            target_value=target_value,
            duration=duration,
            timeout=timeout,
            action_space_info=action_space_info,
        )

    @classmethod
    def from_cfg(cls, cfg: CloseGripperCfg):
        """Create CloseGripper from configuration.

        Note: target_value is hardcoded to -1.0 in __init__, so we don't pass it from cfg.
        """
        return cls(
            agent_assets=cfg.agent_assets,
            duration=cfg.duration,
            timeout=cfg.timeout,
            target_value=cfg.target_value,
            action_space_info=cfg.action_space_info,
        )
