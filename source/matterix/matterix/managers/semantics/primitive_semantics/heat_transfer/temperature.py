# Copyright (c) 2022-2026, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from ...semantics import SemanticState
from ...semantics_cfg import SemanticStateCfg

# Specific heat capacity reference values (J/kg·K)
# Source: https://theengineeringmindset.com/specific-heat-capacity-of-materials/
#
# Lab-relevant materials:
#   Water              4187    Solvents, aqueous solutions
#   Glass               792    Beakers, flasks, vials
#   Aluminium           887    Hot plates, heat blocks, robot links
#   Stainless Steel 316 468    Lab instruments, robot body, IKA plates
#   Cast Iron           554    Heavy lab equipment bases
#   Copper              385    Heating elements, thermocouples
#   Rubber             2005    Septa, gaskets, tubing
#   Ice                2090    Cold baths
#   Salt                881    Salt baths
#   Sand                780    Sand baths


@configclass
class TemperatureCfg(SemanticStateCfg):
    type = "Temperature"
    initial_state: float = 298.15  # Initial temperature (K); default is 25°C
    C: float = 468.0  # Specific heat capacity (J/kg·K); default: Stainless Steel 316
    A: float = 0.1  # Contact/surface area (m²); default: ~10cm × 10cm patch


class Temperature(SemanticState):
    def __init__(self, env, cfg: TemperatureCfg = None, parent_asset=None, name: str = None):
        super().__init__(env, cfg, parent_asset, name)
