# Copyright (c) 2022-2026, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym
import os

from . import (
    test_franka_beaker_lift,
    test_franka_beakers,
    test_franka_vialplate,
    test_franka_rigid_labware_duran_100,
    test_franka_rigid_labware_duran_500,
    test_franka_rigid_labware_falcon_15,
    test_franka_rigid_labware_falcon_50,
    test_franka_rigid_labware_flask_50,
    test_franka_rigid_labware_flask_250,
    test_particle_systems,
    test_semantics_heat_transfer,
)

##
# Register Gym environments.
##

gym.register(
    id="Matterix-Test-Beakers-Franka-v1",
    entry_point="matterix.envs:MatterixBaseEnv",
    kwargs={
        "env_cfg_entry_point": test_franka_beakers.FrankaBeakersEnvTestCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Matterix-Test-Particle-systems-Franka-v1",
    entry_point="matterix.envs:MatterixBaseEnv",
    kwargs={
        "env_cfg_entry_point": test_particle_systems.FrankaBeakersParticleSystemsEnvTestCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Matterix-Test-Beaker-Lift-Franka-v1",
    entry_point="matterix.envs:MatterixBaseEnv",
    kwargs={
        "env_cfg_entry_point": test_franka_beaker_lift.FrankaBeakerLiftEnvTestCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Matterix-Test-Vialplate-Franka-v1",
    entry_point="matterix.envs:MatterixBaseEnv",
    kwargs={
        "env_cfg_entry_point": test_franka_vialplate.FrankaVialplateEnvTestCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Matterix-Test-Semantics-Heat-Transfer-Franka-v1",
    entry_point="matterix.envs:MatterixBaseEnv",
    kwargs={
        "env_cfg_entry_point": test_semantics_heat_transfer.FrankaBeakerHeaterSemanticsEnvTestCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Matterix-Test-Rigid-Labware-Flask-50-Franka-v1",
    entry_point="matterix.envs:MatterixBaseEnv",
    kwargs={
        "env_cfg_entry_point": test_franka_rigid_labware_flask_50.FrankaRigidLabwareFlask50EnvTestCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Matterix-Test-Rigid-Labware-Flask-250-Franka-v1",
    entry_point="matterix.envs:MatterixBaseEnv",
    kwargs={
        "env_cfg_entry_point": test_franka_rigid_labware_flask_250.FrankaRigidLabwareFlask250EnvTestCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Matterix-Test-Rigid-Labware-Duran-100-Franka-v1",
    entry_point="matterix.envs:MatterixBaseEnv",
    kwargs={
        "env_cfg_entry_point": test_franka_rigid_labware_duran_100.FrankaRigidLabwareDuran100EnvTestCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Matterix-Test-Rigid-Labware-Duran-500-Franka-v1",
    entry_point="matterix.envs:MatterixBaseEnv",
    kwargs={
        "env_cfg_entry_point": test_franka_rigid_labware_duran_500.FrankaRigidLabwareDuran500EnvTestCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Matterix-Test-Rigid-Labware-Falcon-15-Franka-v1",
    entry_point="matterix.envs:MatterixBaseEnv",
    kwargs={
        "env_cfg_entry_point": test_franka_rigid_labware_falcon_15.FrankaRigidLabwareFalcon15EnvTestCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Matterix-Test-Rigid-Labware-Falcon-50-Franka-v1",
    entry_point="matterix.envs:MatterixBaseEnv",
    kwargs={
        "env_cfg_entry_point": test_franka_rigid_labware_falcon_50.FrankaRigidLabwareFalcon50EnvTestCfg,
    },
    disable_env_checker=True,
)
