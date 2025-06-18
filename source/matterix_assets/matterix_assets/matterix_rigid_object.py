# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import Dict, List
from isaaclab.utils import configclass

from isaaclab.assets import RigidObjectCfg
@configclass
class MatterixRigidObject(RigidObjectCfg):
    """Configuration parameters for an articulation."""

    frames : Dict[str, List[int]] = {}