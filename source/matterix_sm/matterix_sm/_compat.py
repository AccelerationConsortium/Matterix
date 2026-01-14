# Copyright (c) 2022-2026, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compatibility layer for optional dependencies.

This module provides fallbacks for Isaac Lab dependencies, allowing matterix_sm
to work standalone or with Isaac Lab integration.

When Isaac Lab is available, uses its full-featured configclass.
When unavailable, provides a simplified fallback with basic functionality.
"""

from dataclasses import asdict, dataclass, replace

# Try to import configclass from Isaac Lab, fall back to standard dataclass
try:
    from isaaclab.utils import configclass

    HAS_ISAACLAB = True
    """Flag indicating if Isaac Lab is available."""

except ImportError:
    HAS_ISAACLAB = False
    """Flag indicating if Isaac Lab is available."""

    # Fallback: Use standard Python dataclass with kw_only=True
    # Isaac Lab's configclass allows mixed default/non-default args
    # kw_only=True makes all fields keyword-only, avoiding order restrictions
    def configclass(cls):
        """Simplified drop-in replacement for Isaac Lab's configclass.

        Uses Python's standard dataclass with kw_only=True to allow flexible field ordering.
        Adds basic utility methods for common operations.

        Note: This is a simplified version. For full features (auto type inference,
        mutable field handling, deep validation), install Isaac Lab.

        Features provided:
        - Flexible field ordering (keyword-only args)
        - to_dict() method for dictionary conversion
        - replace() method for creating modified copies
        - copy() method for shallow copies

        Missing compared to Isaac Lab's version:
        - Automatic type inference from default values
        - Automatic default_factory for mutable fields
        - Deep validation of nested configs
        - from_dict() for nested config reconstruction
        """
        # Apply standard dataclass with kw_only
        cls = dataclass(cls, kw_only=True)

        # Add utility methods
        def to_dict(self):
            """Convert config to dictionary."""
            return asdict(self)

        def replace_method(self, **kwargs):
            """Return a new instance with specified fields replaced."""
            return replace(self, **kwargs)

        def copy_method(self):
            """Return a shallow copy of the config."""
            return replace(self)

        # Attach methods to class
        cls.to_dict = to_dict
        cls.replace = replace_method
        cls.copy = copy_method

        return cls


__all__ = ["configclass", "HAS_ISAACLAB"]
