# Copyright (c) 2022-2025, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


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

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


##
# Test Development Environment configs
# Environment with multiple agents (robots), multiple beakers, and tables.
##
@configclass
class FrankaCubeStackEnvTestCfg(MatterixBaseEnvCfg):
    env_spacing = 5.0

    objects = {
        "beaker": BEAKER_500ML_INST_CFG(pos=(0.6, 0.05, 0.05)),
        "table": TABLE_SEATTLE_INST_Cfg(pos=(0.5, 0, 0)),
    }

    articulated_assets = {
        "robot": FRANKA_PANDA_HIGH_PD_IK_CFG(pos=(0.0, 0, 0)),  # robot arm with joint controller
    }

    observations = ObservationManagerCfg()

    record_path = "datasets/dataset.hdf5"
