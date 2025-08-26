# Copyright (c) 2022-2025, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

# Read from environment variable
MATTERIX_PATH = os.getenv("MATTERIX_PATH")

if MATTERIX_PATH is None:
    raise OSError(
        "Environment variable MATTERIX_PATH is not set. Please set it to the root path of the Matterix source."
    )

# Conveniences to other module directories via relative paths
MATTERIX_ASSETS_EXT_DIR = os.path.join(
    MATTERIX_PATH, "source/matterix_assets"
)  # os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
"""Path to the extension source directory."""

MATTERIX_ASSETS_DATA_DIR = os.path.join(MATTERIX_ASSETS_EXT_DIR, "data")
# """Path to the extension data directory."""
