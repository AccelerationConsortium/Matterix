# Copyright (c) 2022-2026, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from typing import Any

from isaaclab.utils import configclass


@configclass
class SemanticsCfg:
    type: str = MISSING
    verbose: bool = False


@configclass
class SemanticStateCfg(SemanticsCfg):
    initial_state: Any = MISSING


@configclass
class SemanticStateTransitionCfg(SemanticsCfg):
    pass


@configclass
class SemanticPredicateCfg(SemanticsCfg):
    pass


@configclass
class semantic_info:
    type: str = MISSING  # predicate type
    asset_name: str = MISSING  # the object the semantic is relevant
    value: Any = MISSING  # the value for the predicate
    additional_info: dict[str, Any] = {}  # other additional information
    env_ids: list[int] = []
