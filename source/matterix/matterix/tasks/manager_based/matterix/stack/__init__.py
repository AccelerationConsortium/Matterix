# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import gymnasium as gym
import os

from . import (
    test_stack
)

##
# Register Gym environments.
##

##
# Joint Position Control
##
gym.register(
    id="Isaac-Stack-Cube-Franka-v1",
    entry_point="matterix.envs:MatterixBaseEnv",
    kwargs={
        "env_cfg_entry_point": test_stack.FrankaCubeStackEnvTestCfg,
    },
    disable_env_checker=True,
)
