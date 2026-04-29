# Copyright (c) 2022-2026, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event terms for randomizing semantic states."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matterix.envs import MatterixBaseEnv


def reset_temperature(
    env: MatterixBaseEnv,
    env_ids: torch.Tensor,
    asset_name: str,
) -> None:
    """Reset temperature of an object to its initial value defined in the semantic config.

    Args:
        env: The environment instance.
        env_ids: Environment indices to reset.
        asset_name: Name of the asset whose temperature to reset.
    """
    if not hasattr(env, "semantic_manager"):
        return

    semantic_name = f"{asset_name}/Temperature"
    if semantic_name not in env.semantic_manager.full_semantic_states:
        return

    env.semantic_manager.full_semantic_states[semantic_name].reset(env_ids)


def randomize_temperature(
    env: MatterixBaseEnv,
    env_ids: torch.Tensor,
    asset_name: str,
    min_temp: float = 273.15,
    max_temp: float = 373.15,
) -> None:
    """Randomize temperature of an object's Temperature semantic.

    Args:
        env: The environment instance.
        env_ids: Environment indices to randomize.
        asset_name: Name of the asset whose temperature to randomize.
        min_temp: Minimum temperature in Kelvin (default: 0°C = 273.15 K).
        max_temp: Maximum temperature in Kelvin (default: 100°C = 373.15 K).
    """
    # Check if semantic manager exists (created after startup events)
    if not hasattr(env, "semantic_manager"):
        return  # Skip if called before semantic manager initialization

    # Build semantic name
    semantic_name = f"{asset_name}/Temperature"

    # Check if semantic exists
    if semantic_name not in env.semantic_manager.full_semantic_states:
        return  # Silently skip if semantic doesn't exist

    # Get temperature semantic
    temperature_semantic = env.semantic_manager.full_semantic_states[semantic_name]

    # Generate random temperatures for specified environments
    random_temps = torch.rand(len(env_ids), device=env.device) * (max_temp - min_temp) + min_temp

    # Update temperature state for specified environments
    temperature_semantic.state[env_ids] = random_temps
