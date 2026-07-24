# Copyright (c) 2022-2026, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
# Configuration for the self-modelled Fisherbrand 03-339-21F reference vial.

from isaaclab.sensors import OffsetCfg
from isaaclab.utils import configclass

from matterix_assets import MATTERIX_ASSETS_DATA_DIR

from ..matterix_rigid_object import MatterixRigidObjectCfg

FISHERBRAND_03_339_21F_DATA_DIR = (
    f"{MATTERIX_ASSETS_DATA_DIR}/labware/fisherbrand_03-339-21f"
)
FISHERBRAND_03_339_21F_USD_PATH = (
    f"{FISHERBRAND_03_339_21F_DATA_DIR}/fisherbrand_03-339-21f_z_up_fixed.usda"
)
FISHERBRAND_03_339_21F_FRAME_OFFSETS = {
    "grasp": OffsetCfg(pos=(0.0, 0.0, 0.0649)),
    "pre_grasp": OffsetCfg(pos=(0.0, 0.0, 0.1649)),
    "post_grasp": OffsetCfg(pos=(0.0, 0.0, 0.1649)),
}


@configclass
class FISHERBRAND_03_339_21F_CFG(MatterixRigidObjectCfg):
    # Dynamic closed Fisherbrand 03-339-21F vial configuration.

    prim_path = "{ENV_REGEX_NS}/RigidObjects_Labware"
    usd_path = FISHERBRAND_03_339_21F_USD_PATH
    scale = (1.0, 1.0, 1.0)
    mass = 0.017734
    activate_contact_sensors = True
    frames = FISHERBRAND_03_339_21F_FRAME_OFFSETS
    semantic_tags = [("class", "vial"), ("asset", "fisherbrand_03-339-21f")]
