# Copyright (c) 2022-2026, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""HeatTransfer semantic preset - bundles temperature and optional convection/conduction."""

from ..primitive_semantics.heat_transfer import HeatConductivityCfg, HeatConvectionCfg, TemperatureCfg
from .semantic_preset import SemanticPreset


class HeatTransferCfg(SemanticPreset):
    """Semantic preset for objects that participate in heat transfer.

    Bundles: Temperature + optional HeatConvection + optional HeatConductivity.

    Note: IsInContactCfg is NOT included here. Add it explicitly to the asset's semantics
    list in the task cfg when physics-derived contact detection is needed.
    HeatConductivity will raise an error at init if no IsInContact is found on the asset.

    Args:
        temp: Initial temperature in Kelvin. Default 298.15 K (25°C).
        C: Specific heat capacity in J/kg·K. Defaults to TemperatureCfg.C if not provided.
        A: Contact/surface area in m². Defaults to TemperatureCfg.A if not provided.
        K: Thermal conductance coefficient in W/m·K. Enables conduction if provided.
        H: Convective heat transfer coefficient in W/m²·K. Enables convection if provided.

    Example:
        beaker = BEAKER_500ML_INST_CFG(
            pos=(0.6, 0.0, 0.0),
            semantics=[
                IsInContactPhysicsCfg(filter_prim_paths_expr=["ika_plate"]),
                HeatTransferCfg(temp=353.15, C=7920.0, A=0.1, K=200),
            ],
        )
    """

    def __init__(
        self,
        temp: float | None = None,
        C: float | None = None,
        A: float | None = None,
        K: float | None = None,
        H: float | None = None,
        verbose: bool = False,
    ):
        super().__init__()

        temp_cfg = TemperatureCfg()
        if temp is not None:
            temp_cfg.initial_state = temp
        if C is not None:
            temp_cfg.C = C
        if A is not None:
            temp_cfg.A = A
        temp_cfg.verbose = verbose
        self.semantics.append(temp_cfg)

        if H is not None:
            self.semantics.append(HeatConvectionCfg(H=H, verbose=verbose))

        if K is not None:
            self.semantics.append(HeatConductivityCfg(K=K, verbose=verbose))
