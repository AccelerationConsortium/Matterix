# Copyright (c) 2022-2026, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Semantic actions that trigger device or state changes without robot motion."""

from .heater_action import TurnOnHeaterCfg

__all__ = [
    "TurnOnHeaterCfg",
]
