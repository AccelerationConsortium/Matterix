# Copyright (c) 2022-2026, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the 500mL beaker.

The following configurations are available:

* :obj:`TABLE_THORLABS_75X90_INST_Cfg`: Instantiated Thorlabs 75x90cm table
* :obj:`TABLE_THORLABS_75X90_Cfg`:  Thorlabs 75x90cm table
* :obj:`TABLE_SEATTLE_INST_Cfg`: Instantiated Seattle lab table
"""

from matterix_assets import MATTERIX_ASSETS_DATA_DIR

from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from ..matterix_static_object import MatterixStaticObjectCfg

##
# Configuration
##

default_prim_path = MatterixStaticObjectCfg().prim_path + "_Infrastructure"


@configclass
class TABLE_THORLABS_75X90_INST_Cfg(MatterixStaticObjectCfg):
    """Properties for the table in the scene."""

    prim_path = default_prim_path  # User-defined object name is appended to the default prim path
    usd_path = f"{MATTERIX_ASSETS_DATA_DIR}/infrastructure/tables/table-thorlabs-75x90/table-inst.usda"
    scale = (1.0, 1.0, 1.0)  # Default scale for the asset

    semantic_tags = [("class", "table")]


@configclass
class TABLE_THORLABS_75X90_Cfg(MatterixStaticObjectCfg):
    """Properties for the table in the scene."""

    prim_path = default_prim_path
    usd_path = f"{MATTERIX_ASSETS_DATA_DIR}/infrastructure/tables/table-thorlabs-75x90/table.usda"
    scale = (1.0, 1.0, 1.0)

    semantic_tags = [("class", "table")]


@configclass
class TABLE_SEATTLE_INST_Cfg(MatterixStaticObjectCfg):
    """Properties for the table in the scene."""

    prim_path = default_prim_path
    usd_path = f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"
    scale = (1.0, 1.0, 1.0)

    rot = (0.707, 0, 0, 0.707)  # 90 degrees around y-axis

    semantic_tags = [("class", "table")]
