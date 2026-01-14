# Copyright (c) 2022-2026, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MoveRelative action - move end-effector by a relative offset."""

from __future__ import annotations

import torch
from typing import ClassVar

from .._compat import configclass
from ..math_utils import quat_mul
from ..robot_action_spaces import ActionSpaceInfo
from ..scene_data import SceneData
from .move_to_pose import MoveToPose, MoveToPoseCfg


@configclass
class MoveRelativeCfg(MoveToPoseCfg):
    """Configuration for MoveRelative action (relative offset from current EE pose).

    Inherits from MoveToPoseCfg and adds offset-specific fields.

    Attributes:
        position_offset: 3D position offset in world frame (x, y, z). Default: no offset.
        orientation_offset: Quaternion offset (w, x, y, z) applied relative to current orientation.
                           Default: identity (no rotation).
        timeout: Defaults to 5.0 seconds.
    """

    position_offset: tuple[float, float, float] | None = None
    orientation_offset: tuple[float, float, float, float] | None = None


class MoveRelative(MoveToPose):
    """Move the end-effector by relative offsets from current pose.

    Inherits from MoveToPose. On first call, computes target as:
    - target_position = current_position + position_offset
    - target_orientation = current_orientation * orientation_offset
    Then behaves like MoveToPose.
    """

    cfg_type: ClassVar[type] = MoveRelativeCfg

    def __init__(
        self,
        agent_assets: str | list[str],
        position_offset: tuple[float, float, float] | None = None,
        orientation_offset: tuple[float, float, float, float] | None = None,
        timeout: float = None,
        position_threshold: float = None,
        orientation_threshold: float = None,
        action_space_info: ActionSpaceInfo | None = None,
    ):
        """
        Args:
            agent_assets: Name(s) of articulated asset(s) acting as agents.
            position_offset: (3,) position offset in world frame (x, y, z).
                            If None, defaults to (0, 0, 0) - no position change.
            orientation_offset: (4,) quaternion offset (w, x, y, z) applied relative to current orientation.
                               If None, defaults to (1, 0, 0, 0) - identity quaternion (no rotation change).
            timeout: Max time (in seconds) before timeout.
            position_threshold: Distance threshold for success (meters).
            orientation_threshold: Orientation threshold for success (radians).
            action_space_info: Optional action space metadata for mask creation.
        """
        # Initialize parent with None for targets (will be set on first call)
        super().__init__(
            agent_assets=agent_assets,
            target_positions_w=None,
            target_orientations_w=None,
            timeout=timeout,
            position_threshold=position_threshold,
            orientation_threshold=orientation_threshold,
            action_space_info=action_space_info,
        )

        # Use defaults if offsets not provided
        # Default position offset: (0, 0, 0) - no translation
        if position_offset is None:
            position_offset = (0.0, 0.0, 0.0)

        # Default orientation offset: (1, 0, 0, 0) - identity quaternion (no rotation)
        if orientation_offset is None:
            orientation_offset = (1.0, 0.0, 0.0, 0.0)

        # Store offsets as tuples (will be converted to tensors in set_execution_params)
        self._position_offset_tuple = position_offset
        self._orientation_offset_tuple = orientation_offset

        # These will be set in set_execution_params()
        self.position_offset = None
        self.orientation_offset = None
        self._target_initialized = False

    def set_execution_params(self, num_envs: int, device: str | torch.device, dt: float) -> None:
        """Set execution parameters and initialize offset tensors."""
        super().set_execution_params(num_envs, device, dt)

        # Convert offset tuples to tensors on the correct device
        self.position_offset = torch.tensor(self._position_offset_tuple, device=self.device, dtype=torch.float32)
        self.orientation_offset = torch.tensor(self._orientation_offset_tuple, device=self.device, dtype=torch.float32)

    def _compute_action_impl(self, scene_data: SceneData, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute relative move action for controlled asset.

        On first call, computes target pose as:
        - target_position = current_position + position_offset
        - target_orientation = current_orientation * orientation_offset (quaternion multiplication)

        Subsequent calls use the cached target (same as MoveToPose).

        Args:
            scene_data: Complete scene state container.
            env_ids: Indices of active environments.

        Returns:
            (action_tensor, action_dim_mask):
                - action_tensor: Shape (num_envs, action_dim) - action values for all envs
                - action_dim_mask: Shape (action_dim,) - which dimensions this action controls
        """
        # On first call, compute target pose relative to current EE pose
        if not self._target_initialized:
            # Get robot articulation data
            if self._asset_name not in scene_data.articulations:
                raise ValueError(
                    f"Asset '{self._asset_name}' not found in scene_data.articulations. "
                    f"Available: {list(scene_data.articulations.keys())}"
                )

            robot_data = scene_data.articulations[self._asset_name]

            if robot_data.ee_pos_w is None:
                raise ValueError(f"End-effector position not available for asset '{self._asset_name}'")
            if robot_data.ee_quat_w is None:
                raise ValueError(f"End-effector orientation not available for asset '{self._asset_name}'")

            # Get current EE pose
            current_pos = robot_data.ee_pos_w.to(self.device)
            current_quat = robot_data.ee_quat_w.to(self.device)

            # Compute target position: current + offset (world frame)
            # Position offset is interpreted in world coordinates - direct vector addition
            # Example: position_offset=(0.1, 0, 0) moves 0.1m along world X-axis
            self.target_positions_w = current_pos + self.position_offset.unsqueeze(0)

            # Compute target orientation: current * offset (local/body frame)
            # Quaternion multiplication applies rotation relative to current orientation
            # Example: 90° pitch offset rotates gripper 90° around its current pitch axis
            # Broadcast orientation offset to all environments
            orientation_offset_broadcast = self.orientation_offset.unsqueeze(0).expand(self.num_envs, -1)
            self.target_orientations_w = quat_mul(current_quat, orientation_offset_broadcast)

            self._target_initialized = True

        # Delegate to parent's implementation
        return super()._compute_action_impl(scene_data, env_ids)

    def _reset_impl(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset target initialization flag when environments are reset."""
        # Target needs to be recomputed relative to new EE pose after reset
        self._target_initialized = False

    @classmethod
    def from_cfg(cls, cfg: MoveRelativeCfg):
        """Create MoveRelative action from configuration."""
        return cls(
            agent_assets=cfg.agent_assets,
            position_offset=cfg.position_offset,
            orientation_offset=cfg.orientation_offset,
            timeout=cfg.timeout,
            position_threshold=cfg.position_threshold,
            orientation_threshold=cfg.orientation_threshold,
            action_space_info=cfg.action_space_info,
        )
