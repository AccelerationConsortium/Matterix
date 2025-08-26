# Copyright (c) 2022-2025, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym
import os

from . import test_franka_beakers

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Test-Beakers-Franka-v1",
    entry_point="matterix.envs:MatterixBaseEnv",
    kwargs={
        "env_cfg_entry_point": test_franka_beakers.FrankaBeakersEnvTestCfg,
    },
    disable_env_checker=True,
)
