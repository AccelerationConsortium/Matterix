# Copyright (c) 2022-2026, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Semantic action information for action-triggered semantic state changes.

This module defines data structures for semantic actions that can be attached to
primitive or compositional actions. When an action completes, it can emit semantic
information that triggers changes in the semantic manager (e.g., temperature changes,
contact state updates, chemical reactions).

Common Semantic Types (examples):

IsHeaterOn:
    SemanticInfo(type="IsHeaterOn", asset_name="heater", value=True,
                additional_info={"target_temperature": 373.15})

IsHeld:
    SemanticInfo(type="IsHeld", asset_name="beaker", value=True)

IsInContact:
    SemanticInfo(type="IsInContact", asset_name="beaker",
                value={"contacting_assets": ["table"]})

Add new semantic types by implementing them in matterix.managers.semantics
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SemanticInfo:
    """Immutable information about a semantic state change triggered by an action.

    Attributes:
        type: Semantic type name (e.g., "Temperature", "IsHeated", "IsInContact").
              Must match a semantic class name in the semantic manager.
        asset_name: Target asset name (e.g., "beaker", "robot").
        value: New value for the semantic state. Type depends on semantic:
            - float: For Temperature, FluidLevel, etc.
            - bool: For IsHeated, IsHeld, etc.
            - dict: For complex semantics like ThermalContact, IsInContact
        additional_info: Optional extra parameters for the semantic.
        env_ids: Specific environments to apply to. If empty, applies to all active environments.

    Example:
        # turn on heater and Set heater temperature to 100°C (373.15 K)
        SemanticInfo(
            type="IsHeaterOn",
            asset_name="heater",
            value=True,
            additional_info={"target_temperature": 373.15})
    """

    type: str
    """Semantic type name matching semantic class in semantic manager."""

    asset_name: str
    """Target asset name."""

    value: Any
    """New value for semantic predicates (type depends on semantic)."""

    additional_info: dict[str, Any] = field(default_factory=dict)
    """Optional extra parameters."""

    env_ids: tuple[int, ...] = field(default_factory=tuple)
    """Specific environments to apply to (empty = all active envs)."""

    def __post_init__(self):
        """Validate semantic info at construction."""
        if not self.type:
            raise ValueError("Semantic type cannot be empty")
        if self.value is None:
            raise ValueError("Semantic value cannot be None")
