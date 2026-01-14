# Copyright (c) 2022-2026, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym
import os

from . import test_franka_beaker_lift, test_franka_beakers, test_particle_systems

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
