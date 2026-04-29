# Copyright (c) 2022-2026, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base class for semantic presets - reusable bundles of primitive semantic configs."""

from ..semantics_cfg import SemanticsCfg


class SemanticPreset:
    """Base class for semantic presets.

    A semantic preset bundles multiple primitive semantic configs into a
    reusable configuration. Pass the result of a preset to an asset's
    `semantics` field.

    Example:
        beaker = BEAKER_500ML_INST_CFG(
            pos=(0.6, 0.0, 0.0),
            semantics=HeatTransferCfg(temp=353.15, C=7920.0, A=0.1),
        )
    """

    def __init__(self):
        self.semantics: list[SemanticsCfg] = []

    def __iter__(self):
        return iter(self.semantics)

    def to_list(self) -> list[SemanticsCfg]:
        return self.semantics
