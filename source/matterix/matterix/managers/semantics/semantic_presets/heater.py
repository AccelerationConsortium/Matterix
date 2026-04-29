# Copyright (c) 2022-2026, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Heater semantic preset - extends HeatTransfer with active heating capability."""

from ..primitive_semantics.heat_transfer import HeatSourceCfg, IsHeaterOnCfg
from .heat_transfer import HeatTransferCfg


class HeaterCfg(HeatTransferCfg):
    """Semantic preset for objects that can actively heat (e.g., hot plates, heater blocks).

    Extends HeatTransferCfg with IsHeaterOn + HeatSource for active heating.

    Args:
        temp: Initial temperature in Kelvin. Defaults to TemperatureCfg.initial_state if not provided.
        C: Specific heat capacity in J/kg·K. REQUIRED.
        A: Contact/surface area in m². REQUIRED.
        K: Thermal conductance coefficient in W/m·K. Enables conduction if provided.
        H: Convective heat transfer coefficient in W/m²·K. Enables convection if provided.
        heater_on: Whether the heater starts active. Defaults to IsHeaterOnCfg.default_value if not provided.
        target_temperature: Target temperature the heater drives toward in Kelvin.
            Defaults to IsHeaterOnCfg.target_temperature if not provided.
        K_heater: Heater rate constant in sec⁻¹. Defaults to HeatSourceCfg.K if not provided.

    Example:
        hot_plate = IKA_PLATE_INST_CFG(
            pos=(0.4, -0.3, 0.12),
            semantics=HeaterCfg(temp=298.15, C=500.0, A=0.5, heater_on=True, target_temperature=373.15),
        )
    """

    def __init__(
        self,
        temp: float | None = None,
        C: float | None = None,
        A: float | None = None,
        K: float | None = None,
        H: float | None = None,
        heater_on: bool | None = None,
        target_temperature: float | None = None,
        K_heater: float | None = None,
        verbose: bool = False,
    ):
        super().__init__(temp=temp, C=C, A=A, K=K, H=H, verbose=verbose)
        # Build IsHeaterOnCfg - all defaults come from IsHeaterOnCfg itself
        is_heater_on_cfg = IsHeaterOnCfg()
        if heater_on is not None:
            is_heater_on_cfg.default_value = heater_on
        if target_temperature is not None:
            is_heater_on_cfg.target_temperature = target_temperature
        is_heater_on_cfg.verbose = verbose
        # Build HeatSourceCfg - K defaults come from HeatSourceCfg itself
        heat_source_cfg = HeatSourceCfg()
        if K_heater is not None:
            heat_source_cfg.K = K_heater
        heat_source_cfg.verbose = verbose
        self.semantics += [is_heater_on_cfg, heat_source_cfg]
