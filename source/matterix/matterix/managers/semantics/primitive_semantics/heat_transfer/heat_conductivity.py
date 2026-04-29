# Copyright (c) 2022-2026, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from dataclasses import MISSING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.utils import configclass

from ...primitive_semantics import IsInContact
from ...semantics import ElementNotFoundError, SemanticStateTransition
from ...semantics_cfg import SemanticStateTransitionCfg
from .temperature import Temperature


@configclass
class HeatConductivityCfg(SemanticStateTransitionCfg):
    type = "HeatConductivity"
    K: float = MISSING  # thermal conductance coefficient, unit: W/m·K
    # Link to conductance coefficient of different materials:
    # http://hyperphysics.phy-astr.gsu.edu/hbase/Tables/thrcn.html


class HeatConductivity(SemanticStateTransition):
    def __init__(self, env, cfg: HeatConductivityCfg = None, parent_asset=None, name: str = None):
        super().__init__(env, cfg, parent_asset, name)

        self.temperature: Temperature = None
        self.contacted_obj_temprature: Temperature = None
        self.precond_is_in_contact: IsInContact = None

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
                if isinstance(self.env.scene[asset_name], (RigidObject, Articulation)):
                    try:
                        self.contacted_obj_temprature = self.find_corresponding_semantics(Temperature, asset_name)
                    except ElementNotFoundError:
                        print(
                            f"[HeatConductivity] WARNING: '{asset_name}' has no Temperature semantic — skipping heat"
                            " transfer."
                        )
                        continue
                    self.print_status(f"contacted object temperature={self.contacted_obj_temprature.state}")

                    Q_dot = (
                        self.cfg.K
                        * self.temperature.cfg.A
                        * (self.temperature.state - self.contacted_obj_temprature.state)
                    )
                    delta = (-Q_dot * self.dt) / (self.parent_asset.cfg.mass * self.temperature.cfg.C)

                    # Only apply to envs where contact is active
                    self.rate_of_change = torch.where(in_contact, delta, self.rate_of_change)

        # add here to the object rate of change
        self.temperature.add_rate_of_change(self.rate_of_change)

        self.print_status(f"rate_of_change={self.rate_of_change}")
