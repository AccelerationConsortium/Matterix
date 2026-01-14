# Copyright (c) 2022-2026, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base class and configuration for all actions."""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, ClassVar

# Use compatibility layer for optional Isaac Lab dependency
from ._compat import configclass

if TYPE_CHECKING:
    pass


@configclass
class ActionBaseCfg:
    """Base configuration for all actions (primitive and compositional).

    This is the root of the configuration hierarchy. All action configs inherit from this.
    Naming matches ActionBase for consistency.

    Attributes:
        description: Optional human-readable description of what this action does.
                    Useful for documentation and workflow definitions.
    """

    description: str = ""
    """Optional description of what this action does."""


class ActionBase(ABC):
    """Abstract base class for all actions (primitive and compositional).

    This provides a common interface and factory method for creating actions from configs.

    Subclasses should define:
        cfg_type: ClassVar[type] - The config class for this action type
    """

    # Subclasses must override this to specify their config type
    cfg_type: ClassVar[type] = None

    @classmethod
    def _build_config_registry(cls):
        """Build a registry mapping config types to action classes.

        Automatically discovers all action subclasses and their cfg_type attributes.

        Returns:
            Dict mapping config class to action class.
        """
        registry = {}

        def collect_subclasses(base_class):
            """Recursively collect all subclasses."""
            for subclass in base_class.__subclasses__():
                if hasattr(subclass, "cfg_type") and subclass.cfg_type is not None:
                    registry[subclass.cfg_type] = subclass
                # Recursively process nested subclasses
                collect_subclasses(subclass)

        collect_subclasses(cls)
        return registry

    @classmethod
    def from_cfg(cls, cfg: ActionBaseCfg):
        """Generic factory to create any action from config.

        This method automatically dispatches to the correct action class based on config type.
        Action classes self-register by defining their cfg_type class variable.

        Args:
            cfg: Action configuration (any subclass of ActionBaseCfg).

        Returns:
            Action instance of the appropriate type.

        Example:
            >>> from matterix.state_machine import ActionBase, PickObjectCfg
            >>> cfg = PickObjectCfg(object="beaker", asset="robot", device="cuda", num_envs=16)
            >>> action = ActionBase.from_cfg(cfg)
            >>> # Returns PickObject instance automatically!
        """
        # Import all action modules to ensure they're loaded for registry
        from . import compositional_actions  # noqa: F401
        from . import primitive_actions  # noqa: F401

        # Build registry dynamically from all loaded action classes
        config_registry = cls._build_config_registry()

        cfg_type = type(cfg)

        # First try exact match
        if cfg_type in config_registry:
            action_class = config_registry[cfg_type]
            return action_class.from_cfg(cfg)

        # If no exact match, check inheritance (for configs that inherit from registered types)
        # This allows PickObjectCfg (inherits CompositionalActionCfg) to use CompositionalAction
        for registered_cfg_type, registered_action_class in config_registry.items():
            if isinstance(cfg, registered_cfg_type):
                return registered_action_class.from_cfg(cfg)

        # No match found
        available_types = [ct.__name__ for ct in config_registry.keys()]
        raise ValueError(
            f"Unknown config type: {cfg_type.__name__}. Cannot create action. "
            f"Available config types: {available_types}\n"
            "Did you forget to set the cfg_type class variable in your action class?"
        )
