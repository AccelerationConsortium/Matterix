# Copyright (c) 2022-2026, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration and data paths for the promoted 3_5 vial holder."""

import json
from pathlib import Path

from isaaclab.sensors import OffsetCfg
from isaaclab.utils import configclass

from matterix_assets import MATTERIX_ASSETS_DATA_DIR

from ..matterix_rigid_object import MatterixRigidObjectCfg

VIALPLATE_3_5_DATA_DIR = f"{MATTERIX_ASSETS_DATA_DIR}/labware/vialplate_3_5"
VIALPLATE_3_5_USD_PATH = f"{VIALPLATE_3_5_DATA_DIR}/3_5_vialplate_free_standing_frames.usda"
VIALPLATE_3_5_FRAME_CONTRACT_PATH = f"{VIALPLATE_3_5_DATA_DIR}/holder-hole-frame-contract.json"


def _load_holder_hole_frames() -> dict[str, OffsetCfg]:
    """Load the public 15-hole frame family from the promoted contract."""
    contract = json.loads(Path(VIALPLATE_3_5_FRAME_CONTRACT_PATH).read_text(encoding="utf-8"))
    frames = contract.get("frames")
    if not isinstance(frames, list) or len(frames) != 15:
        raise ValueError("promoted vial-holder frame contract must contain exactly 15 frames")
    orientation = tuple(float(value) for value in contract["frame_orientation_wxyz"])
    return {
        frame["name"]: OffsetCfg(pos=tuple(float(value) for value in frame["position_m"]), rot=orientation)
        for frame in frames
    }


VIALPLATE_3_5_HOLE_FRAMES = _load_holder_hole_frames()


@configclass
class VIALPLATE_3_5_CFG(MatterixRigidObjectCfg):
    """Dynamic 15-well holder with the promoted frame-bearing USD payload."""

    prim_path = "{ENV_REGEX_NS}/RigidObjects_Labware"
    usd_path = VIALPLATE_3_5_USD_PATH
    scale = (1.0, 1.0, 1.0)
    mass = 0.070350472
    activate_contact_sensors = True
    frames = VIALPLATE_3_5_HOLE_FRAMES
    semantic_tags = [("class", "vial_holder"), ("asset", "3_5_vialplate")]
