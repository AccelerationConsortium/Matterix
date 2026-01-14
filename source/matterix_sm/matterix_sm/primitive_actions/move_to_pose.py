# Copyright (c) 2022-2026, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MoveToPose action - move end-effector to target world pose (position + orientation)."""

from __future__ import annotations

import torch
from typing import ClassVar

from .._compat import configclass
from ..primitive_action import PrimitiveAction, PrimitiveActionCfg
from ..robot_action_spaces import ActionSpaceInfo
from ..scene_data import SceneData


@configclass
class MoveToPoseCfg(PrimitiveActionCfg):
    """Configuration for MoveToPose action (move to target pose).

    Inherits base fields (assets, timeout) from PrimitiveActionCfg.

    Attributes:
        target_positions: Target positions in world frame, shape (num_envs, 3).
                         If None, uses current robot position (position-only hold).
        target_orientations: Target orientation quaternions in world frame, shape (num_envs, 4) as (w,x,y,z).
                            If None, uses current robot orientation (orientation-only hold).
        position_threshold: Distance threshold for success (meters).
        orientation_threshold: Orientation threshold for success (radians).
        settling_time: Time (in seconds) the robot must remain within threshold before success.
                      Prevents false positives from overshoots or oscillations.
    """

    target_positions: torch.Tensor | None = None
    target_orientations: torch.Tensor | None = None
    position_threshold: float = 0.01  # 10mm tolerance
    orientation_threshold: float = 0.02  # ~1.15 degrees tolerance (realistic for IK)
    settling_time: float = 0.05  # 50ms default (tunable per task)


class MoveToPose(PrimitiveAction):
    """Move the end-effector to target world pose (position + orientation).

    Controls position and orientation only, does not modify gripper state.

    Supports partial specification:
    - target_positions_w=None: Hold current position (orientation-only control)
    - target_orientations_w=None: Hold current orientation (position-only control)
    - Both specified: Full pose control (default behavior)
    """

    cfg_type: ClassVar[type] = MoveToPoseCfg

    def __init__(
        self,
        agent_assets: str | list[str],
        target_positions_w: torch.Tensor | None,
        target_orientations_w: torch.Tensor | None,
        timeout: float,
        position_threshold: float,
        orientation_threshold: float,
        settling_time: float = 0.05,
        action_space_info: ActionSpaceInfo | None = None,
    ):
        """
        Args:
            agent_assets: Name(s) of articulated asset(s) acting as agents.
            target_positions_w: (num_envs, 3) world-frame target positions.
                               If None, uses current robot position on first call.
            target_orientations_w: (num_envs, 4) world-frame target orientations as quaternions (w,x,y,z).
                                  If None, uses current robot orientation on first call.
            timeout: Max time (in seconds) before timeout.
            position_threshold: Distance threshold for success (meters).
            orientation_threshold: Orientation threshold for success (radians).
            settling_time: Time (in seconds) the robot must remain within threshold before success.
            action_space_info: Optional action space metadata for mask creation.
        """
        super().__init__(agent_assets, timeout, action_space_info)

        # Track whether targets were originally None (for proper reset behavior)
        self._position_was_none = target_positions_w is None
        self._orientation_was_none = target_orientations_w is None

        # Store target tensors (will be moved to device in set_execution_params)
        self._target_positions_w_init = target_positions_w
        self._target_orientations_w_init = target_orientations_w

        # These will be set in set_execution_params()
        self.target_positions_w = None
        self.target_orientations_w = None

        # Initialize flag for lazy target initialization
        self._targets_initialized = False

        self.position_threshold = position_threshold
        self.orientation_threshold = orientation_threshold
        self.settling_time = settling_time

        # Settling time tracking (initialized in set_execution_params)
        self.time_in_threshold = None

        # Validate action_space_info at init (fail-fast)
        if self.action_space_info is None:
            raise ValueError(
                "MoveToPose requires action_space_info to determine position/orientation indices. "
                "Pass action_space_info parameter when creating the action."
            )

        # Cache indices for fast access
        self._position_indices = (
            list(self.action_space_info.position_indices) if self.action_space_info.position_indices is not None else []
        )
        self._orientation_indices = (
            list(self.action_space_info.orientation_indices)
            if self.action_space_info.orientation_indices is not None
            else []
        )

        # These will be initialized in set_execution_params()
        self._action_dim_mask = None
        self._action_tensor = None

    def set_execution_params(self, num_envs: int, device: str | torch.device, dt: float) -> None:
        """Set execution parameters and initialize move-specific tensors."""
        super().set_execution_params(num_envs, device, dt)

        # Validate and move target tensors to device
        if self._target_positions_w_init is not None:
            assert self._target_positions_w_init.shape == (
                num_envs,
                3,
            ), f"target_positions_w must have shape (num_envs, 3), got {self._target_positions_w_init.shape}"
            self.target_positions_w = self._target_positions_w_init.to(self.device)
        else:
            self.target_positions_w = None

        if self._target_orientations_w_init is not None:
            assert self._target_orientations_w_init.shape == (
                num_envs,
                4,
            ), f"target_orientations_w must have shape (num_envs, 4), got {self._target_orientations_w_init.shape}"
            self.target_orientations_w = self._target_orientations_w_init.to(self.device)
        else:
            self.target_orientations_w = None

        # Cache the action dimension mask (computed once, reused every step)
        self._action_dim_mask = self._create_action_mask("position_orientation")

        # Pre-allocate action tensor (will be zeroed and reused each step)
        self._action_tensor = torch.zeros((self.num_envs, self.action_space_info.total_dim), device=self.device)

        # Initialize settling time tracker
        self.time_in_threshold = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        # Cache asset name (single-agent for now)
        self._asset_name = self.agent_assets[0]

    def _compute_action_impl(self, scene_data: SceneData, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute move to pose action for controlled asset.

        Args:
            scene_data: Complete scene state container.
            env_ids: Indices of active environments currently executing this action.

        Returns:
            (action_tensor, action_dim_mask):
                - action_tensor: Shape (num_envs, action_dim) - action values for all envs
                - action_dim_mask: Shape (action_dim,) - which dimensions this action controls
        """
        import time

        compute_start = time.perf_counter()
        timings = {}

        # Get robot articulation data (use cached asset name)
        data_access_start = time.perf_counter()
        if self._asset_name not in scene_data.articulations:
            raise ValueError(
                f"Asset '{self._asset_name}' not found in scene_data.articulations. "
                f"Available: {list(scene_data.articulations.keys())}"
            )

        robot_data = scene_data.articulations[self._asset_name]
        timings["data_access"] = time.perf_counter() - data_access_start

        # On first call, fill in None values with current robot pose
        # This allows partial specification (e.g., move position only, keep current orientation)
        init_start = time.perf_counter()
        if not hasattr(self, "_targets_initialized") or not self._targets_initialized:
            if self.target_positions_w is None:
                # Use current EE position if not specified
                if robot_data.ee_pos_w is None:
                    raise ValueError(f"End-effector position not available for asset '{self._asset_name}'")
                self.target_positions_w = robot_data.ee_pos_w.to(self.device).clone()

            if self.target_orientations_w is None:
                # Use current EE orientation if not specified
                if robot_data.ee_quat_w is None:
                    raise ValueError(f"End-effector orientation not available for asset '{self._asset_name}'")
                self.target_orientations_w = robot_data.ee_quat_w.to(self.device).clone()

            self._targets_initialized = True
        timings["target_init"] = time.perf_counter() - init_start

        # Zero the pre-allocated action tensor (reuse memory)
        zero_start = time.perf_counter()
        self._action_tensor.zero_()
        timings["tensor_zero"] = time.perf_counter() - zero_start

        # Convert target pose (position + orientation) from world frame to robot base frame
        # Target is specified in world frame, action is in robot base frame
        frame_convert_start = time.perf_counter()
        target_positions_b, target_orientations_b = self._convert_world_to_base_frame(
            scene_data,
            self._asset_name,
            self.target_positions_w,
            self.target_orientations_w,
        )
        timings["frame_conversion"] = time.perf_counter() - frame_convert_start

        # Set position dimensions using cached indices (only for active environments)
        tensor_fill_start = time.perf_counter()
        for i, idx in enumerate(self._position_indices):
            self._action_tensor[env_ids, idx] = target_positions_b[env_ids, i]

        # Set orientation dimensions using cached indices (only for active environments)
        for i, idx in enumerate(self._orientation_indices):
            self._action_tensor[env_ids, idx] = target_orientations_b[env_ids, i]
        timings["tensor_fill"] = time.perf_counter() - tensor_fill_start

        # Note: Gripper state is NOT modified by this action (controlled by mask)

        total_time = time.perf_counter() - compute_start

        # Print timing breakdown if significantly slow (>5ms)
        if total_time > 0.005:
            RED = "\033[91m"
            YELLOW = "\033[93m"
            RESET = "\033[0m"
            print(f"\n{RED}{'─' * 80}")
            print(f"⚠️  SLOW ACTION COMPUTE: {self.__class__.__name__} took {total_time * 1000:.2f}ms")
            print(f"{'─' * 80}{RESET}")
            print(f"{YELLOW}  Action timing breakdown:{RESET}")
            sorted_timings = sorted(timings.items(), key=lambda x: x[1], reverse=True)
            for name, timing_val in sorted_timings:
                timing_ms = timing_val * 1000
                percent = (timing_val / total_time) * 100 if total_time > 0 else 0
                print(f"{YELLOW}    ▸ {name}: {timing_ms:.2f}ms ({percent:.1f}%){RESET}")
            print(f"{RED}{'─' * 80}{RESET}\n")

        return self._action_tensor, self._action_dim_mask

    def _check_completion_impl(self, scene_data: SceneData, env_ids: torch.Tensor) -> None:
        """Check if target pose reached (both position and orientation within thresholds).

        Args:
            scene_data: Complete scene state container.
            env_ids: Indices of active environments.
        """
        # Skip checking if targets not initialized yet
        # This happens when targets were None and haven't been filled from robot state yet
        if (
            not hasattr(self, "target_positions_w")
            or not hasattr(self, "target_orientations_w")
            or self.target_positions_w is None
            or self.target_orientations_w is None
        ):
            self._env_success_mask[env_ids] = False
            self._env_failure_mask[env_ids] = False
            return

        # Get robot articulation data
        if self._asset_name not in scene_data.articulations:
            raise ValueError(
                f"Asset '{self._asset_name}' not found in scene_data.articulations. "
                f"Available: {list(scene_data.articulations.keys())}"
            )

        robot_data = scene_data.articulations[self._asset_name]

        # Validate end-effector data is available
        if robot_data.ee_pos_w is None:
            raise ValueError(f"End-effector position not available for asset '{self._asset_name}'")
        if robot_data.ee_quat_w is None:
            raise ValueError(f"End-effector orientation not available for asset '{self._asset_name}'")

        # Position distance check
        current_pos = robot_data.ee_pos_w.to(self.device)
        position_distance = torch.norm(self.target_positions_w - current_pos, dim=1)
        position_reached = position_distance < self.position_threshold

        # Orientation distance check using quaternion dot product
        # Formula: angular_distance = 2 * arccos(|dot(q1, q2)|)
        # This gives the geodesic distance on SO(3) in radians
        # The abs() handles quaternion double cover (q and -q represent same rotation)
        current_quat = robot_data.ee_quat_w.to(self.device)
        dot_product = torch.sum(self.target_orientations_w * current_quat, dim=1)
        # Clamp to [0, 1] to avoid numerical issues with arccos
        dot_product_clamped = torch.clamp(torch.abs(dot_product), 0.0, 1.0)
        angular_distance = 2.0 * torch.acos(dot_product_clamped)
        orientation_reached = angular_distance < self.orientation_threshold

        # Check if both position and orientation are within thresholds
        within_threshold = position_reached & orientation_reached

        # Update settling timer for active environments
        # If within threshold: accumulate time, else reset to zero
        self.time_in_threshold[env_ids] = torch.where(
            within_threshold[env_ids],
            self.time_in_threshold[env_ids] + self.dt,  # Accumulate
            torch.zeros_like(self.time_in_threshold[env_ids]),  # Reset
        )

        # Success only if settled for required duration
        self._env_success_mask[env_ids] = self.time_in_threshold[env_ids] >= self.settling_time

        # No custom failure modes
        self._env_failure_mask[env_ids] = False

    def _reset_impl(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset target initialization flag and settling timers when environments are reset.

        This ensures that if targets were None, they'll be re-initialized
        from the new post-reset robot pose.

        Args:
            env_ids: Indices of environments being reset, or None for all.
        """
        # Reset target initialization flag so targets are recomputed from new pose
        self._targets_initialized = False

        # Reset settling timers for the specified environments
        if env_ids is None:
            # Reset all environments
            self.time_in_threshold.zero_()
        else:
            # Reset specific environments
            self.time_in_threshold[env_ids] = 0.0

        # If targets were originally None, restore them to None
        # This ensures they'll be re-filled from the new robot pose after reset
        if self._position_was_none:
            self.target_positions_w = None

        if self._orientation_was_none:
            self.target_orientations_w = None

    @classmethod
    def from_cfg(cls, cfg: MoveToPoseCfg):
        """Create MoveToPose action from configuration."""
        return cls(
            agent_assets=cfg.agent_assets,
            target_positions_w=cfg.target_positions,
            target_orientations_w=cfg.target_orientations,
            timeout=cfg.timeout,
            position_threshold=cfg.position_threshold,
            orientation_threshold=cfg.orientation_threshold,
            settling_time=cfg.settling_time,
            action_space_info=cfg.action_space_info,
        )
