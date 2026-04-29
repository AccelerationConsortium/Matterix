# Copyright (c) 2022-2026, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Semantic presets - reusable bundles of primitive semantic configs.

Presets compose multiple primitive semantics into a single convenient configuration.
Pass preset instances directly to an asset's `semantics` field.

Available presets:
    HeatTransferCfg: Temperature + contact + optional convection/conduction
    HeaterCfg:       HeatTransfer + active heating (IsHeaterOn + HeatSource)
"""

from .heat_transfer import HeatTransferCfg
from .heater import HeaterCfg
from .semantic_preset import SemanticPreset

__all__ = [
    "SemanticPreset",
    "HeatTransferCfg",
    "HeaterCfg",
]
