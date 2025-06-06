# Copyright (c) 2022-2025, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Package containing asset and sensor configurations."""

import os
import toml

# Conveniences to other module directories via relative paths
MATTERIX_ASSETS_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
"""Path to the extension source directory."""

MATTERIX_ASSETS_DATA_DIR = os.path.join(MATTERIX_ASSETS_EXT_DIR, "data")
"""Path to the extension data directory."""

MATTERIX_ASSETS_METADATA = toml.load(os.path.join(MATTERIX_ASSETS_EXT_DIR, "config", "extension.toml"))
"""Extension metadata dictionary parsed from the extension.toml file."""

# Configure the module-level variables
__version__ = MATTERIX_ASSETS_METADATA["package"]["version"]
