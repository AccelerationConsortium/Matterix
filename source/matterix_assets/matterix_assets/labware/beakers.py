# Copyright (c) 2022-2025, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the 500mL beaker.

The following configurations are available:

* :obj:`BEAKER_500ML_INST_CFG`: Instantiated 500mL borosilicate glass beaker
* :obj:`BEAKER_500ML_CFG`: 500mL borosilicate glass beaker

"""

from matterix_assets import MATTERIX_ASSETS_DATA_DIR

from isaaclab.utils import configclass

from ..matterix_rigid_object import MatterixRigidObjectCfg

##
# Configuration
##

default_prim_path = MatterixRigidObjectCfg().prim_path + "_Labware"


@configclass
class BEAKER_500ML_INST_CFG(MatterixRigidObjectCfg):
    """Properties for the beaker to manipulate in the scene."""

    prim_path = default_prim_path  # User-defined object name is appended to the default prim path
    usd_path = f"{MATTERIX_ASSETS_DATA_DIR}/labware/beaker500ml/beaker-500ml-inst.usda"

    scale = (1.0, 1.0, 1.0)  # Default scale for the beaker
    mass = 0.3  # Mass of the beaker in kg
    activate_contact_sensors = False  # Activate contact sensors for the beaker

    frames = {
        "pre_grasp": (0.0, 0.0, 0.04),
        "grasp": (0.0, 0.0, 0.0),
        "post_grasp": (0.0, 0.0, 0.05),
    }
    semantic_tags = [("class", "beaker")]


@configclass
class BEAKER_500ML_CFG(MatterixRigidObjectCfg):
    """Properties for the beaker to manipulate in the scene."""

    prim_path = default_prim_path  # User-defined object name is appended to the default prim path
    usd_path = f"{MATTERIX_ASSETS_DATA_DIR}/labware/beaker500ml/beaker-500ml.usda"

    scale = (1.0, 1.0, 1.0)  # Default scale for the beaker
    mass = 0.3  # Mass of the beaker in kg
    activate_contact_sensors = False  # Activate contact sensors for the beaker

    frames = {
        "pre_grasp": (0.0, 0.0, 0.04),
        "grasp": (0.0, 0.0, 0.0),
        "post_grasp": (0.0, 0.0, 0.05),
    }
    semantic_tags = [("class", "beaker")]
