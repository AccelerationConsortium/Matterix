# Copyright (c) 2022-2025, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configs for liquid and powder environments"""

from matterix.envs import LightStateCfg, MatterixBaseEnvCfg
from matterix.particle_systems import FluidCfg, PowderCfg, ReservedFluidCfg, ReservedPowderCfg
from matterix_assets.infrastructure.tables import TABLE_SEATTLE_INST_Cfg
from matterix_assets.labware import BEAKER_500ML_CFG
from matterix_assets.robots import FRANKA_PANDA_HIGH_PD_CFG

from isaaclab.sim.spawners.lights import DomeLightCfg, SphereLightCfg
from isaaclab.utils import configclass

##
# Pre-defined configs
##


@configclass
class FrankaBeakersParticleSystemsEnvTestCfg(MatterixBaseEnvCfg):
    # lights
    lights = {
        "light1": LightStateCfg(
            light=SphereLightCfg(
                color=(1.0, 1.0, 1.0), intensity=8e5, enable_color_temperature=True, color_temperature=5500
            ),
            pos=(5, 0, 5),
        ),
        "light2": LightStateCfg(light=DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=1000.0)),
    }

    # assets
    objects = {
        # Set each stacking cube deterministically
        "table": TABLE_SEATTLE_INST_Cfg(pos=(0.5, 0, 0)),
        "beaker": BEAKER_500ML_CFG(pos=(0.6, 0.1, 0.03)),
        "beaker_2": BEAKER_500ML_CFG(pos=(0.6, -0.1, 0.03)),
    }

    articulated_assets = {
        "robot": FRANKA_PANDA_HIGH_PD_CFG(),
    }

    record_path = "datasets/dataset.hdf5"

    particle_systems = {
        "test_fluid": FluidCfg(pos=(0.6, 0.1, 0.1), volume=(0.01, 0.01, 0.2)),
        "test_powder": PowderCfg(pos=(0.6, -0.1, 0.1), volume=(0.01, 0.01, 0.2)),
    }
    reserved_particle_systems = [
        ReservedFluidCfg(num_reserved_particle_sys=1),
        ReservedPowderCfg(num_reserved_particle_sys=1),
    ]

    episode_length_s = 5.0
