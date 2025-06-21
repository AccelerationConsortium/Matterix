# Copyright (c) 2022-2025, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Package containing asset and sensor configurations."""

import os
import toml

from .constants import MATTERIX_ASSETS_DATA_DIR, MATTERIX_ASSETS_EXT_DIR
from .matterix_articulation import MatterixArticulationCfg
from .matterix_rigid_object import MatterixRigidObject
from .robots import *
