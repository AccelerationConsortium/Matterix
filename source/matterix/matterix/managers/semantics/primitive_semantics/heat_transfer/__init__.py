# Copyright (c) 2022-2026, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from .heat_conductivity import HeatConductivity, HeatConductivityCfg
from .heat_convection import AmbientAirHeatConvection, AmbientAirHeatConvectionCfg, HeatConvection, HeatConvectionCfg
from .heat_source import HeatSource, HeatSourceCfg
from .is_heater_on import IsHeaterOn, IsHeaterOnCfg
from .temperature import Temperature, TemperatureCfg

__semantics_cfg__ = [
    "TemperatureCfg",
    "HeatSourceCfg",
    "HeatConductivityCfg",
    "IsHeaterOnCfg",
    "HeatConvectionCfg",
    "AmbientAirHeatConvectionCfg",
]

__semantics__ = [
    "Temperature",
    "HeatSource",
    "HeatConductivity",
    "IsHeaterOn",
    "HeatConvection",
    "AmbientAirHeatConvection",
]

__all__ = __semantics_cfg__ + __semantics__

# Create a dictionary mapping class names to class objects
heat_transfer_semantics_dict = {}
for class_name in __semantics__:
    class_obj = globals()[class_name]
    heat_transfer_semantics_dict[class_name] = class_obj
