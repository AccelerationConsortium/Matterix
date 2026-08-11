# Copyright (c) 2022-2026, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""LOCAL-ONLY configs for the six Batch 1 labware checkpoint candidates.

These configs are test-environment plumbing. The reusable USD payloads live in
the Matterix_assets data submodule; the capped payloads are spawned as generic
USD scene assets by their dedicated checkpoint environments because each has
two rigid bodies joined by a fixed joint.
"""

import json
import os

from matterix_assets import MATTERIX_ASSETS_DATA_DIR
from matterix.managers.semantics.primitive_semantics import IsInContactPhysicsCfg

from isaaclab.utils import configclass

from ..matterix_rigid_object import MatterixRigidObjectCfg


default_prim_path = "{ENV_REGEX_NS}/RigidObjects_Labware"

CORNING_4980_50_MASS_KG = 0.0305
CORNING_4980_250_MASS_KG = 0.1122


def _load_frames(slug: str) -> dict[str, tuple[float, float, float]]:
    """Load authored interface frames from the staged payload."""
    path = os.path.join(MATTERIX_ASSETS_DATA_DIR, "labware", slug, "frames.json")
    with open(path) as handle:
        contract = json.load(handle)
    frames = contract.get("frames")
    if not isinstance(frames, dict) or not frames:
        raise ValueError(f"{path} carries no frames")
    missing = {"grasp", "pre_grasp", "post_grasp"} - set(frames)
    if missing:
        raise ValueError(f"{path} is missing required frames: {sorted(missing)}")
    return {name: tuple(offset) for name, offset in frames.items()}


@configclass
class CORNING_4980_50_LOCAL_ONLY_CFG(MatterixRigidObjectCfg):
    """Corning PYREX 4980-50, 50 mL narrow-mouth Erlenmeyer flask."""

    prim_path = default_prim_path
    usd_path = f"{MATTERIX_ASSETS_DATA_DIR}/labware/corning-4980-50/corning-4980-50-inst.usda"
    scale = (1.0, 1.0, 1.0)
    mass = CORNING_4980_50_MASS_KG
    activate_contact_sensors = True
    frames = _load_frames("corning-4980-50")
    semantic_tags = [("class", "flask")]
    semantics = [
        IsInContactPhysicsCfg(
            filter_prim_paths_expr=["robot/panda_leftfinger", "robot/panda_rightfinger"]
        )
    ]


@configclass
class CORNING_4980_250_LOCAL_ONLY_CFG(MatterixRigidObjectCfg):
    """Corning PYREX 4980-250, 250 mL narrow-mouth Erlenmeyer flask."""

    prim_path = default_prim_path
    usd_path = f"{MATTERIX_ASSETS_DATA_DIR}/labware/corning-4980-250/corning-4980-250-inst.usda"
    scale = (1.0, 1.0, 1.0)
    mass = CORNING_4980_250_MASS_KG
    activate_contact_sensors = True
    frames = _load_frames("corning-4980-250")
    semantic_tags = [("class", "flask")]
    semantics = [
        IsInContactPhysicsCfg(
            filter_prim_paths_expr=["robot/panda_leftfinger", "robot/panda_rightfinger"]
        )
    ]
