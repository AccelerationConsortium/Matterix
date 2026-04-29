# Copyright (c) 2022-2026, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pure semantic action that triggers semantic state changes instantly."""

import torch
from dataclasses import MISSING

from .._compat import configclass
from ..primitive_action import PrimitiveAction, PrimitiveActionCfg
from ..scene_data import SceneData
from ..semantic_info import SemanticInfo


@configclass
class SemanticActionCfg(PrimitiveActionCfg):
    """Configuration for pure semantic action (no motion).

    This action completes immediately and emits semantic state changes.
    It does not control any robot (agent_assets is empty by default).

    Attributes:
        agent_assets: Empty by default (no robots controlled). Leave as default.
        semantics: REQUIRED - list of semantic state changes to emit.

    Example:
        turn_on_heater = SemanticActionCfg(
            semantics=[
                SemanticInfo(type="IsHeaterOn", asset_name="heater", value=True)
            ]
        )
    """

    agent_assets: str | list[str] = []  # Empty - no agents controlled
    semantics: list[SemanticInfo] = MISSING  # REQUIRED for SemanticAction

    def __post_init__(self):
        super().__post_init__()


class SemanticAction(PrimitiveAction):
    """Pure semantic action that completes immediately.

    This action performs no motion - it just emits semantic state changes
    when executed. Useful for:
    - Turn on/off devices (heaters, pumps, etc.)
    - Set temperature targets
    - Mark objects as held/released
    - Toggle fluid flow

    The action succeeds immediately on first step and emits its semantics.

    Example:
        # Turn on heater
        turn_on_heater = SemanticAction.from_cfg(SemanticActionCfg(
            semantics=[
                SemanticInfo(type="IsHeaterOn",
                            asset_name="heater",
                            value=True)
            ]
        ))

        # Use in action sequence
        sm.set_action_sequence([
            MoveToFrame(...),
            turn_on_heater,  # ← Instant semantic change
            Wait(duration=5.0),
        ])
    """

    cfg_type = SemanticActionCfg

    def __init__(
        self,
        agent_assets: str | list[str],
        timeout: float,
        action_space_info=None,
        semantics: list | None = None,
    ):
        """Initialize semantic action.

        Args:
            agent_assets: Dummy value (not used).
            timeout: Max time before timeout (not really used since action completes immediately).
            action_space_info: Optional action space info.
            semantics: List of semantic state changes to emit.
        """
        super().__init__(agent_assets, timeout, action_space_info, semantics)

        if not semantics:
            raise ValueError("SemanticAction requires at least one SemanticInfo in semantics list")

    def _compute_action_impl(self, scene_data: SceneData, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute no-op action (dummy return values).

        Since agent_assets=[], StateMachine doesn't iterate over this action's
        agents, so these return values are never actually used. We return minimal
        tensors just to satisfy the interface contract.

        Args:
            scene_data: Scene state.
            env_ids: Active environment indices.

        Returns:
            (action_tensor, action_mask): Minimal dummy tensors.
                These values are never used since agent_assets=[] means
                StateMachine skips the action accumulation loop entirely.
        """
        # Return minimal tensors - values don't matter since agent_assets=[] means
        # StateMachine never uses them (loop over agent_assets doesn't execute)
        action_tensor = torch.zeros(self.num_envs, 1, device=self.device)
        action_mask = torch.zeros(1, dtype=torch.bool, device=self.device)

        return action_tensor, action_mask

    def _check_completion_impl(self, scene_data: SceneData, env_ids: torch.Tensor) -> None:
        """Mark all environments as successful immediately.

        Args:
            scene_data: Scene state.
            env_ids: Active environment indices.
        """
        self._env_success_mask[env_ids] = True
