# Copyright (c) 2022-2026, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Semantic actions for heater control."""

from dataclasses import MISSING

from .._compat import configclass
from ..primitive_actions.semantic_action import SemanticActionCfg
from ..semantic_info import SemanticInfo


@configclass
class TurnOnHeaterCfg(SemanticActionCfg):
    """Turn a heater asset on or off and optionally set its target temperature.

    Convenience wrapper around SemanticActionCfg for the IsHeaterOn predicate.
    No robot motion — completes instantly and emits the semantic change.

    Attributes:
        asset_name: Name of the heater asset (e.g., "ika_plate").
        value: True to turn on, False to turn off.
        target_temperature: Optional setpoint in Kelvin. If None, existing
            target_temperature on the heater is kept unchanged.

    Example::

        workflows = {
            "heat_beaker": [
                PlaceObjectCfg(...),
                TurnOnHeaterCfg(
                    heater_asset_name="ika_plate",
                    value=True,
                    target_temperature=373.15,
                ),
                WaitCfg(duration=60.0),
                TurnOnHeaterCfg(heater_asset_name="ika_plate", value=False),
            ]
        }
    """

    asset_name: str = MISSING
    value: bool = MISSING
    target_temperature: float | None = None
    semantics: list | None = None  # built from asset_name/value/target_temperature at from_cfg time

    def build_semantics(self) -> list[SemanticInfo]:
        """Build SemanticInfo list from this cfg's fields. Called by SemanticAction.from_cfg."""
        additional_info = {}
        if self.target_temperature is not None:
            additional_info["target_temperature"] = self.target_temperature
        return [
            SemanticInfo(
                type="IsHeaterOn",
                asset_name=self.asset_name,
                value=self.value,
                additional_info=additional_info,
            )
        ]
