# Copyright (c) 2022-2025, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the 500mL beaker.

The following configurations are available:

* :obj:`BEAKER_500ML_INST_CFG`: Instantiated 500mL borosilicate glass beaker
* :obj:`BEAKER_500ML_CFG`: 500mL borosilicate glass beaker

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
class BEAKER_500ML_INST_CFG(MatterixRigidObjectCfg):
    """Properties for the beaker to manipulate in the scene."""

    prim_path += "/Labware"
    init_state = RigidObjectCfg.InitialStateCfg(pos=(0.6, 0.1, 0.05))
    spawn = sim_utils.UsdFileCfg(
        usd_path=f"{MATTERIX_ASSETS_DATA_DIR}/labware/beaker500ml/beaker-500ml-inst.usda",
        scale=(1.0, 1.0, 1.0),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.3),  # kg
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        physics_material=sim_utils.RigidBodyMaterialCfg(),
        activate_contact_sensors=False,
    )


@configclass
class BEAKER_500ML_CFG(MatterixRigidObjectCfg):
    """Properties for the beaker to manipulate in the scene."""

    prim_path += "/Labware"
    init_state = RigidObjectCfg.InitialStateCfg(pos=(0.6, 0.1, 0.05))
    spawn = sim_utils.UsdFileCfg(
        usd_path=f"{MATTERIX_ASSETS_DATA_DIR}/labware/beaker500ml/beaker-500ml.usda",
        scale=(1.0, 1.0, 1.0),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.3),  # kg
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        physics_material=sim_utils.RigidBodyMaterialCfg(),
        activate_contact_sensors=False,
    )
