# Copyright (c) 2022-2026, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test development environment with multiple Franka robots and beakers."""

from matterix.envs import MatterixBaseEnvCfg, mdp
from matterix_assets.infrastructure.tables import TABLE_SEATTLE_INST_Cfg, TABLE_THORLABS_75X90_INST_Cfg
from matterix_assets.labware.beakers import BEAKER_500ML_INST_CFG
from matterix_assets.robots import FRANKA_PANDA_HIGH_PD_CFG, FRANKA_PANDA_HIGH_PD_IK_CFG

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass


##
# Observation configs
##
@configclass
class ObservationManagerCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        ee_pos_robot = ObsTerm(func=mdp.ee_env_pos, params={"asset_name": "robot"})
        ee_pos_robot2 = ObsTerm(func=mdp.ee_env_pos, params={"asset_name": "robot2"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


##
# Test Development Environment configs
# Environment with multiple agents (robots), multiple beakers, and tables.
##
@configclass
class FrankaBeakersEnvTestCfg(MatterixBaseEnvCfg):
    env_spacing = 5.0

    objects = {
        "beaker": BEAKER_500ML_INST_CFG(pos=(0.6, 0.05, 0.05)),
        "beaker2": BEAKER_500ML_INST_CFG(pos=(0.6, -0.05, 0.05)),
        "beaker3": BEAKER_500ML_INST_CFG(pos=(0.6, 1.95, -0.2)),
        "beaker4": BEAKER_500ML_INST_CFG(pos=(0.6, 2.05, -0.2)),
        "beaker5": BEAKER_500ML_INST_CFG(pos=(0.55, 2.0, -0.2)),
        "table": TABLE_SEATTLE_INST_Cfg(pos=(0.5, 0, 0)),
        "table2": TABLE_THORLABS_75X90_INST_Cfg(pos=(0.0, 2, -0.25)),
    }

    articulated_assets = {
        "robot": FRANKA_PANDA_HIGH_PD_CFG(pos=(0.0, 0, 0)),  # robot arm with joint controller
        "robot2": FRANKA_PANDA_HIGH_PD_IK_CFG(
            pos=(0, 2, -0.25)
        ),  # robot arm with Cartesian space controller using IK solver
    }

    observations = ObservationManagerCfg()

    record_path = "datasets/dataset.hdf5"
