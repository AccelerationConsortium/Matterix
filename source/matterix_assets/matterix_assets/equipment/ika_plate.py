# Copyright (c) 2022-2026, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the IKA plate.

The following configurations are available:

* :obj:`IKA_PLATE_INST_CFG`: Instantiated IKA plate

"""

from matterix_assets import MATTERIX_ASSETS_DATA_DIR

from isaaclab.utils import configclass

from ..matterix_rigid_object import MatterixRigidObjectCfg

##
# Configuration
##

default_prim_path = MatterixRigidObjectCfg().prim_path + "_Equipment"


@configclass
class IKA_PLATE_INST_CFG(MatterixRigidObjectCfg):
    """Properties for the IKA plate to manipulate in the scene."""

    prim_path = default_prim_path  # User-defined object name is appended to the default prim path
    usd_path = f"{MATTERIX_ASSETS_DATA_DIR}/equipment/balance-heater-stirrer/IKA-plate-inst.usda"

    scale = (1.0, 1.0, 1.0)  # Default scale for the beaker
    # scale = (0.8, 0.8, 0.8)  # Default scale for the beaker
    mass = 1.0  # Mass of the beaker in kg
    activate_contact_sensors = False  # Activate contact sensors for the beaker

    frames = {
        "pre_place": (0.0, 0.0, 0.25),  # approach height above plate surface
        "place": (0.0, 0.0, 0.13),  # resting height on plate surface (beaker bottom)
        "post_place": (0.0, 0.0, 0.25),  # retreat height after releasing
    }
    semantic_tags = [("class", "balance")]
