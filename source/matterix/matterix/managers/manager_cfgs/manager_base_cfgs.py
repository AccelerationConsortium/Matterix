# Copyright (c) 2022-2025, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""


@configclass
class EventCfg:
    """Configuration for events."""


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""
