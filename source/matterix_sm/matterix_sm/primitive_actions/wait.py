# Copyright (c) 2022-2026, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wait action that holds the current robot state for a specified duration."""

import torch
from dataclasses import MISSING

from .._compat import configclass
from ..primitive_action import PrimitiveAction, PrimitiveActionCfg
from ..scene_data import SceneData


@configclass
class WaitCfg(PrimitiveActionCfg):
    """Configuration for wait action.

    The wait action pauses execution for a fixed duration without moving the robot.
    The robot holds its last commanded position (StateMachine preserves action_dict).

    Attributes:
        agent_assets: Empty by default (no robots controlled). Leave as default.
        duration: Time in seconds to wait before succeeding.

    Example:
        # Wait 3 seconds after placing beaker on heater
        wait = WaitCfg(duration=3.0)

        sm.set_action_sequence([
            MoveToFrame(...),
            CloseGripper(...),
            SemanticAction(...),  # Turn on heater
            WaitCfg(duration=5.0),  # Wait for heating
        ])
    """

    agent_assets: str | list[str] = []  # Empty - no agents controlled
    duration: float = MISSING  # Required: seconds to wait
    timeout: float = float("inf")  # Wait always runs to completion — no timeout


class Wait(PrimitiveAction):
    """Wait action that holds state for a fixed duration then succeeds.

    The robot does not move - the StateMachine preserves whatever action_dict
    values were set by the previous action (since agent_assets=[] means
    StateMachine skips the action accumulation loop entirely).

    Each environment independently tracks its elapsed wait time, so parallel
    environments that start waiting at different steps complete independently.

    Example:
        sm.set_action_sequence([
            MoveToFrame(..., action_space_info=FRANKA_IK_ACTION_SPACE),
            CloseGripper(..., action_space_info=FRANKA_IK_ACTION_SPACE),
            Wait(duration=2.0),  # Hold for 2 seconds
        ])
    """

    cfg_type = WaitCfg

    def __init__(
        self,
        agent_assets: str | list[str],
        timeout: float,
        action_space_info=None,
        semantics=None,
        duration: float = 1.0,
    ):
        """Initialize wait action.

        Args:
            agent_assets: Unused (always empty).
            timeout: Max time before timeout. Should be >= duration.
            action_space_info: Optional action space info (unused).
            semantics: Optional semantic actions to emit on success.
            duration: Seconds to wait before succeeding.
        """
        super().__init__(agent_assets, timeout, action_space_info, semantics)
        self.duration = duration

    def _compute_action_impl(self, scene_data: SceneData, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return dummy tensors (robot holds last commanded state).

        Since agent_assets=[], StateMachine never uses these values.
        The robot continues executing its last commanded action unchanged.
        """
        action_tensor = torch.zeros(self.num_envs, 1, device=self.device)
        action_mask = torch.zeros(1, dtype=torch.bool, device=self.device)
        return action_tensor, action_mask

    def _check_completion_impl(self, scene_data: SceneData, env_ids: torch.Tensor) -> None:
        """Mark environments as successful once duration has elapsed.

        Note: time_elapsed is incremented by the base class before this is called.
        Each env independently tracks its own elapsed time.
        """
        elapsed = self.time_elapsed[env_ids]
        done = elapsed >= self.duration
        if done.any():
            self._env_success_mask[env_ids[done]] = True

    @classmethod
    def from_cfg(cls, cfg: WaitCfg) -> "Wait":
        """Create Wait from WaitCfg.

        Args:
            cfg: Wait configuration.

        Returns:
            Wait action instance.
        """
        return cls(
            agent_assets=cfg.agent_assets,
            timeout=cfg.timeout,
            action_space_info=cfg.action_space_info,
            semantics=cfg.semantics,
            duration=cfg.duration,
        )
