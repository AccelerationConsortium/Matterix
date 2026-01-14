# Copyright (c) 2022-2026, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base class for compositional actions - sequences of primitive or compositional actions."""

from __future__ import annotations

import torch
from dataclasses import MISSING
from typing import ClassVar

from ._compat import configclass
from .action_base import ActionBase, ActionBaseCfg
from .scene_data import SceneData


@configclass
class CompositionalActionCfg(ActionBaseCfg):
    """Base configuration for compositional actions.

    Compositional actions are sequences of sub-actions, which can be either:
    - Primitive actions (Move, MoveToFrame, OpenGripper, etc.)
    - Other compositional actions (enabling multi-level hierarchies)

    This enables unlimited nesting depth for complex hierarchical workflows.

    Attributes:
        sub_actions: List of ActionBaseCfg (primitive or compositional) that define the sequence.
                     StateMachine will automatically inject num_envs/device/timeout into
                     primitive sub-actions before instantiation.

    Note:
        Compositional actions don't need num_envs, device, or timeout because they don't
        execute directly - they decompose into primitives. The StateMachine injects these
        parameters into primitive sub-actions automatically.

    Example - Multi-level hierarchy:
        CompositionalActionCfg(
            sub_actions=[
                PickObjectCfg(...),  # Compositional
                MoveToFrameCfg(...),  # Primitive
                CompositionalActionCfg(  # Nested compositional!
                    sub_actions=[
                        OpenGripperCfg(...),
                        MoveRelativeCfg(...),
                    ]
                ),
            ]
        )
    """

    sub_actions: list[ActionBaseCfg] = MISSING


class CompositionalAction(ActionBase):
    """Base class for compositional actions.

    Compositional actions are sequences of sub-actions that can be either primitive
    or other compositional actions, enabling unlimited nesting depth.

    All configuration should be done at construction/config time. No runtime
    initialization from environment is required with SceneData approach.

    Usage:
        Config-based (data-driven):
        - Provide sub_action_configs in constructor or config
        - Actions created automatically from configs
        - Supports unlimited nesting
    """

    cfg_type: ClassVar[type] = CompositionalActionCfg

    def __init__(
        self,
        # assets: str | list[str],
        # device: str,
        # timeout: int,
        sub_action_configs: list[ActionBaseCfg] | None = None,
    ):
        """
        Args:
            assets: Name(s) of asset(s) to control. REQUIRED.
            device: Device string ("cuda" or "cpu"). REQUIRED.
            timeout: Max steps before timeout.
            sub_action_configs: List of ActionBaseCfg that define the sequence.
        """
        super().__init__()

        # Normalize to list format
        # self.assets = [assets] if isinstance(assets, str) else assets
        # self.device = torch.device(device)
        # self.timeout = timeout
        self.sub_action_configs = sub_action_configs or []

        # Create actions from configs immediately
        self.actions_list = []
        if self.sub_action_configs:
            self._initialize_from_configs()

    def _initialize_from_configs(self):
        """Create actions from sub_action_configs using the factory pattern.

        Recursively initializes nested compositional actions.
        """
        from .action_base import ActionBase

        self.actions_list = []
        for cfg in self.sub_action_configs:
            # Use factory to create actions from configs
            action = ActionBase.from_cfg(cfg)
            self.actions_list.append(action)

    def compute_action(
        self, scene_data: SceneData, env_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Execute the current sub-action in the sequence.

        Args:
            scene_data: Complete scene state container.
            env_ids: Indices of active environments.

        Returns:
            (action_tensor, success_mask, timeout_mask)
        """
        # For compositional actions, we need to track which sub-action each environment is on
        # This is managed by StateMachine's flattened action list, so we just pass through
        # to all sub-actions with the same env_ids
        raise NotImplementedError(
            "CompositionalAction.compute_action() should not be called directly. "
            "StateMachine flattens compositional actions into primitive sequences."
        )

    def reset(self) -> None:
        """Reset all sub-actions."""
        for action in self.actions_list:
            if hasattr(action, "reset"):
                action.reset()

    @classmethod
    def from_cfg(cls, cfg: CompositionalActionCfg):
        """Create CompositionalAction from configuration."""
        return cls(
            # assets=cfg.assets,
            # device=cfg.device,
            # timeout=cfg.timeout,
            sub_action_configs=cfg.sub_actions,
        )
