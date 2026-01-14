# Copyright (c) 2022-2026, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Global constants for state machine actions.

This module contains only truly global constants that are used across multiple actions.
Action-specific values should be defined in their respective config classes.
"""

from __future__ import annotations

# Time-based defaults (in seconds)
TIMEOUT_DEFAULT = 10.0
"""Default timeout (in seconds) for actions that do not specify their own.
10 seconds is reasonable for most manipulation actions."""


# Standard end-effector orientation (quaternion w, x, y, z)
# This represents a 90° pitch rotation (gripper pointing downward)
FRANKA_ORIENTATION_DOWN_QUAT = (0.0, 0.7071, 0.7071, 0.0)
