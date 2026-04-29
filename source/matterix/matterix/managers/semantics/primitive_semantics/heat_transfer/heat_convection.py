# Copyright (c) 2022-2026, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from dataclasses import MISSING

from matterix.particle_systems import FluidSystem

from isaaclab.utils import configclass

from ...primitive_semantics import IsInContact
from ...semantics import SemanticStateTransition
from ...semantics_cfg import SemanticStateTransitionCfg
from .temperature import Temperature


@configclass
class HeatConvectionCfg(SemanticStateTransitionCfg):
    type = "HeatConvection"
    H: float = MISSING  # convective heat transfer coefficient, unit: W/m²·K
    # convection coefficients:
    # https://www.engineersedge.com/heat_transfer/convective_heat_transfer_coefficients__13378.htm
    # Water:  Free convection: 500-1000 W/m²·K, Forced: 1000-10000 W/m²·K
    # Air:    Free convection: 5-25 W/m²·K,     Forced: 10-200 W/m²·K
    # Oil:    Free convection: 50-100 W/m²·K,   Forced: 100-1000 W/m²·K


@configclass
class AmbientAirHeatConvectionCfg(HeatConvectionCfg):
    type = "AmbientAirHeatConvection"
    H: float = 25  # W/m²·K - free convection in air
    ambient_temperature: float = 298.15  # Ambient air temperature (K); default is 25°C


class HeatConvection(SemanticStateTransition):
    def __init__(self, env, cfg: HeatConvectionCfg = None, parent_asset=None, name: str = None):
        super().__init__(env, cfg, parent_asset, name)

        self.temperature: Temperature = None
        self.contacted_obj_temprature: Temperature = None
        self.precond_is_in_contact: IsInContact = None  # immersed in fluid/air

    def init(self):
        # check if the object has the Temperature state
        self.temperature = self.find_corresponding_semantics(Temperature)
        self.precond_is_in_contact = self.find_corresponding_semantics(IsInContact)
        self.rate_of_change = torch.zeros_like(self.temperature.state, dtype=torch.float32, device=self.env.device)

    def step(self):
        # Compute the rate of heat transfer
        self.rate_of_change = torch.zeros_like(self.temperature.state, dtype=torch.float32, device=self.env.device)

        in_contact = self.precond_is_in_contact.is_in_contact  # (num_envs,)
        if in_contact.any():
            for asset_name in self.precond_is_in_contact.contacting_assets:
                if isinstance(self.env.scene[asset_name], FluidSystem):
                    self.contacted_obj_temprature = self.find_corresponding_semantics(Temperature, asset_name)
                    self.print_status(f"contacted fluid temperature={self.contacted_obj_temprature.state}")

                    Q_dot = (
                        -self.cfg.H
                        * self.temperature.cfg.A
                        * (self.temperature.state - self.contacted_obj_temprature.state)
                    )
                    delta = (Q_dot * self.dt) / (self.parent_asset.cfg.mass * self.temperature.cfg.C)

                    self.rate_of_change = torch.where(in_contact, delta, self.rate_of_change)

        # add here to the object rate of change
        self.temperature.add_rate_of_change(self.rate_of_change)

        self.print_status(f"rate_of_change={self.rate_of_change}")


class AmbientAirHeatConvection(HeatConvection):
    cfg: AmbientAirHeatConvectionCfg

    def __init__(self, env, cfg: AmbientAirHeatConvectionCfg = None, parent_asset=None, name: str = None):
        super().__init__(env, cfg, parent_asset, name)

        self.temperature: Temperature = None
        self.all_obj_with_temperature: list[Temperature] = None

    def init(self):
        self.rate_of_change = torch.full((self.env.num_envs,), 0.0, dtype=torch.float32, device=self.env.device)
        self.all_obj_with_temperature = self.find_all_corresponding_semantics(Temperature)

    def step(self):
        # Compute the rate of heat transfer
        for obj_temperature in self.all_obj_with_temperature:
            Q_dot = -self.cfg.H * obj_temperature.cfg.A * (obj_temperature.state - self.cfg.ambient_temperature)

            # Get mass from config
            asset_name = obj_temperature.name.split("/")[0]
            if hasattr(obj_temperature.parent_asset, "cfg") and hasattr(obj_temperature.parent_asset.cfg, "mass"):
                # RigidObject - has runtime cfg with mass
                mass = obj_temperature.parent_asset.cfg.mass
            elif hasattr(self.env.cfg, "objects") and asset_name in self.env.cfg.objects:
                # Static object - look up from env.cfg.objects
                mass = self.env.cfg.objects[asset_name].mass
            elif hasattr(self.env.cfg, "articulated_assets") and asset_name in self.env.cfg.articulated_assets:
                # Articulation - look up from env.cfg.articulated_assets
                mass = self.env.cfg.articulated_assets[asset_name].mass
            else:
                raise ValueError(f"Cannot find mass for asset '{asset_name}'")

            self.rate_of_change = (Q_dot * self.dt) / (mass * obj_temperature.cfg.C)
            obj_temperature.add_rate_of_change(self.rate_of_change)
            self.print_status(f"rate_of_change={self.rate_of_change}, object={obj_temperature.name}")
