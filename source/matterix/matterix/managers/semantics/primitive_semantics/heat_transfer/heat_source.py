# Copyright (c) 2022-2026, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

from isaaclab.utils import configclass

from ...semantics import SemanticStateTransition
from ...semantics_cfg import SemanticStateTransitionCfg
from .temperature import Temperature


@configclass
class HeatSourceCfg(SemanticStateTransitionCfg):
    type = "HeatSource"
    K: float = 0.05  # unit: (sec)^{-1}


class HeatSource(SemanticStateTransition):
    def __init__(self, env, cfg: HeatSourceCfg = None, parent_asset=None, name: str = None):
        super().__init__(env, cfg, parent_asset, name)
        self.rate_of_change: torch.Tensor = None

    def init(self):
        self.temperature = self.find_corresponding_semantics(Temperature)
        self.rate_of_change = torch.zeros_like(self.temperature.state, dtype=torch.float32, device=self.env.device)
        # Import here to avoid circular import (is_heater_on imports from heat_source)
        from .is_heater_on import IsHeaterOn

        self.heater_predicate = self.find_corresponding_semantics(IsHeaterOn)

    def step(self):
        # Per-env: only apply heat where heater is on
        self.rate_of_change = torch.where(
            self.heater_predicate.is_heater_on,
            self.cfg.K * self.dt * (self.heater_predicate.target_temperature - self.temperature.state),
            torch.zeros_like(self.temperature.state),
        )
        self.temperature.add_rate_of_change(self.rate_of_change)
        self.print_status(f"rate_of_change={self.rate_of_change}")
