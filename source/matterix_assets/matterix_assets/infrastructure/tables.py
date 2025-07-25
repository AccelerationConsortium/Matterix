# Copyright (c) 2022-2025, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the 500mL beaker.

The following configurations are available:

* :obj:`TABLE_THORLABS_75X90_INST_Cfg`: Instantiated Thorlabs 75x90cm table
* :obj:`TABLE_THORLABS_75X90_Cfg`:  Thorlabs 75x90cm table

"""

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass

from matterix_assets import MATTERIX_ASSETS_DATA_DIR
from ..matterix_rigid_object import MatterixRigidObjectCfg

##
# Configuration
##


@configclass
class TABLE_THORLABS_75X90_INST_Cfg(MatterixRigidObjectCfg):
    """Properties for the table in the scene."""

    prim_path += "/Infrastructure"
    translation = (0.0, 0.0, 0.0)  # The translation to apply to the prim w.r.t. its parent prim.
    orientation = (1.0, 0.0, 0.0, 0.0)  # The orientation in (w, x, y, z) to apply to the prim w.r.t. its parent prim.
    spawn = sim_utils.UsdFileCfg(
        usd_path=f"{MATTERIX_ASSETS_DATA_DIR}/infrastructure/tables/table-thorlabs-75x90/table-inst.usda",
        scale=(1.0, 1.0, 1.0),
    )


@configclass
class TABLE_THORLABS_75X90_Cfg(MatterixRigidObjectCfg):
    """Properties for the table in the scene."""

    prim_path += "/Infrastructure"
    translation = (0.0, 0.0, 0.0)
    orientation = (1.0, 0.0, 0.0, 0.0)
    spawn = sim_utils.UsdFileCfg(
        usd_path=f"{MATTERIX_ASSETS_DATA_DIR}/infrastructure/tables/table-thorlabs-75x90/table.usda",
        scale=(1.0, 1.0, 1.0),
    )
