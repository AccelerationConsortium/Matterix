# Copyright (c) 2022-2026, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from typing import Any

from .semantics_cfg import (
    SemanticPredicateCfg,
    SemanticsCfg,
    SemanticStateCfg,
    SemanticStateTransitionCfg,
    semantic_info,
)


class ElementNotFoundError(Exception):
    """Custom exception raised when an element is not found in the list."""

    pass


class Semantics:
    """Base class for handling semantics.
    The Semantic are added to an object with the following functions
    1. step
    2. reset
    3. initialize
    These functions will be called by the manager that the semantic is added to.
    Note: This is a abstract class and should not be used on it's own.
    """

    def __init__(self, env, cfg: SemanticsCfg, parent_asset=None, name: str = None):
        """
        Initializes a semantic object with the given environment.
        Args:
            env: The environment for the semantic object.
        """
        self.env = env
        self.cfg = cfg
        self.parent_asset = parent_asset
        self.name = name

        if self.__class__.__name__ != self.cfg.type:
            raise ValueError(
                f"The semantics that is called is {self.__class__.__name__} while the cfg file is related to another"
                f" semantics {self.cfg.type}"
            )

    def init(self):
        """
        Initializes the semantic objects.
        """
        pass

    def reset(self, env_ids):
        """
        Resets the semantic object with the given environment IDs.
        Args:
            env_ids: The environment IDs.
        """
        self.print_status(f"resetting env_ids={env_ids}")

    def step(self):
        """
        steps the semantic object.
        """
        pass

    def get_value(self) -> Any:
        """
        Gets the state of the semantic object.
        Returns:
            The state of the semantic object.
        """
        pass

    def set_value(self, semantic_val: semantic_info):
        """
        Sets the value of a precondition of the semantic object from a semantic action.
        Args:
            semantic_val: a semantic variable of the semantic object set by a semantic action.
                          It might be different types: torch.Tensor, bool, float, int, dict.
        """
        pass

    def print_status(self, msg: str) -> None:
        if self.cfg.verbose:
            print(f"[{self.name}] {msg}")

    def find_corresponding_semantics(self, state_type, related_asset_name: str = None):
        asset = self.parent_asset
        if related_asset_name is not None:
            asset = self.env.scene[related_asset_name]

        for state in asset.semantic_list:
            if isinstance(state, state_type):
                return state
        asset_label = related_asset_name if related_asset_name is not None else type(asset).__name__
        raise ElementNotFoundError(
            f"No element of type {state_type.__name__} found in the semantic list of asset {asset_label}."
        )

    def find_all_corresponding_state(self, state_type):
        full_semantic_list = []
        for semantic in self.env.semantic_manager.full_semantic_states.values():
            if isinstance(semantic, state_type):
                full_semantic_list.append(semantic)
        return full_semantic_list

    def find_all_corresponding_state_transition(self, state_type):
        full_semantic_list = []
        for semantic in self.env.semantic_manager.full_semantic_state_transitions.values():
            if isinstance(semantic, state_type):
                full_semantic_list.append(semantic)
        return full_semantic_list

    def find_all_corresponding_predicates(self, state_type):
        full_semantic_list = []
        for semantic in self.env.semantic_manager.full_semantic_predicates.values():
            if isinstance(semantic, state_type):
                full_semantic_list.append(semantic)
        return full_semantic_list

    def find_all_corresponding_semantics(self, state_type):
        full_semantic_list = []
        full_semantic_list.extend(self.find_all_corresponding_state(state_type))
        full_semantic_list.extend(self.find_all_corresponding_state_transition(state_type))
        full_semantic_list.extend(self.find_all_corresponding_predicates(state_type))

        if full_semantic_list:
            return full_semantic_list
        raise ElementNotFoundError(
            f"No element of type {state_type.__name__} found in the semantic predicate list of env"
            f" {self.env.semantic_manager.full_semantic_predicates.keys()}."
        )


class SemanticState(Semantics):
    """Base class for handling semantic states."""

    def __init__(self, env, cfg: SemanticStateCfg = None, parent_asset=None, name: str = None):
        super().__init__(env, cfg, parent_asset, name)
        self.state: torch.Tensor = torch.full(
            (self.env.num_envs,),
            self.cfg.initial_state,
            dtype=torch.float32,
            device=self.env.device,
        )
        self.rate_of_change: torch.Tensor = torch.zeros_like(self.state)

    def init(self):
        pass

    def reset(self, env_ids):
        super().reset(env_ids)
        self.state[env_ids] = self.cfg.initial_state  # type: ignore
        self.rate_of_change[env_ids] = torch.zeros_like(self.rate_of_change[env_ids])  # type: ignore
        self.print_status(f"state={self.state}, rate_of_change={self.rate_of_change}")

    def step(self):
        if self.rate_of_change is not None:
            self.state += self.rate_of_change
            self.rate_of_change.zero_()
        self.print_status(f"state={self.state}")

    def get_state(self) -> torch.Tensor:
        return self.state

    def add_rate_of_change(self, rate_of_change: torch.Tensor):
        self.rate_of_change += rate_of_change


class SemanticStateTransition(Semantics):
    """Base class for handling semantic state transition."""

    def __init__(self, env, cfg: SemanticStateTransitionCfg, parent_asset=None, name: str = None):
        super().__init__(env, cfg, parent_asset, name)
        self.dt = self.env.physics_dt * self.env.cfg.decimation
        self.rate_of_change: torch.Tensor = None

    def init(self):
        """
        Initializes the semantic objects.
        """
        pass

    def reset(self, env_ids):
        super().reset(env_ids)
        self.rate_of_change[env_ids] = torch.zeros_like(self.rate_of_change[env_ids])  # type: ignore
        self.print_status(f"reset env_ids={env_ids}, rate_of_change={self.rate_of_change}")

    def get_value(self) -> torch.Tensor:
        """
        Gets the rate of change (state transition) associated to the class semantic object.
        Returns:
            The state of the semantic object.
        """
        return self.rate_of_change

    def set_value(self, semantic_val: Any):
        """
        Sets the value of a precondition of the semantic object from a semantic action.
        Args:
            semantic_val: a semantic variable of the semantic object set by a semantic action.
                          It might be different types: torch.Tensor, bool, float, int, dict.
        """
        pass


class SemanticPredicate(Semantics):
    """Base class for handling semantic Predicates."""

    def __init__(self, env, cfg: SemanticPredicateCfg, parent_asset=None, name: str = None):
        """
        Initializes a semantic object with the given environment.
        Args:
            env: The environment for the semantic object.
        """
        super().__init__(env, cfg, parent_asset, name)

    def set_value(self, semantic_val: semantic_info):
        print("semantic name:", self.name, "semantic info:", semantic_info)
