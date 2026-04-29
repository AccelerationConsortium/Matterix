# Copyright (c) 2022-2026, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

from isaaclab.utils import configclass

from ...semantics import SemanticPredicate
from ...semantics_cfg import SemanticPredicateCfg, semantic_info
from .temperature import Temperature


@configclass
class IsHeaterOnCfg(SemanticPredicateCfg):
    type = "IsHeaterOn"
    default_value: bool = False
    target_temperature: float = 298.15  # Target temperature in Kelvin


class IsHeaterOn(SemanticPredicate):
    def __init__(self, env, cfg: IsHeaterOnCfg = None, parent_asset=None, name: str = None):
        super().__init__(env, cfg, parent_asset, name)
        self.is_heater_on: torch.Tensor = torch.full(
            (self.env.num_envs,), cfg.default_value, dtype=torch.bool, device=self.env.device
        )
        self.target_temperature: torch.Tensor = torch.full(
            (self.env.num_envs,), cfg.target_temperature, dtype=torch.float32, device=self.env.device
        )

    def init(self):
        self.temperature = self.find_corresponding_semantics(Temperature)

    def set_value(self, semantic_info: semantic_info):
        """Sets the heater state and optionally overrides target temperature."""
        # Empty env_ids tuple means apply to all environments
        ids = slice(None) if len(semantic_info.env_ids) == 0 else list(semantic_info.env_ids)
        self.is_heater_on[ids] = semantic_info.value
        if (
            "target_temperature" in semantic_info.additional_info
            and semantic_info.additional_info["target_temperature"] is not None
        ):
            self.target_temperature[ids] = semantic_info.additional_info["target_temperature"]

    def reset(self, env_ids):
        super().reset(env_ids)
        self.is_heater_on[env_ids] = self.cfg.default_value
        self.target_temperature[env_ids] = self.cfg.target_temperature

    def step(self):
        self.print_status(f"is_heater_on={self.is_heater_on}, target_temperature={self.target_temperature}")
