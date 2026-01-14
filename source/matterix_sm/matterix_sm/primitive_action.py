# Copyright (c) 2022-2026, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base class for primitive actions."""

from __future__ import annotations

import torch
from abc import abstractmethod
from dataclasses import MISSING

from ._compat import configclass
from .action_base import ActionBase, ActionBaseCfg
from .action_constants import TIMEOUT_DEFAULT
from .math_utils import quat_mul, quat_rotate
from .robot_action_spaces import ActionSpaceInfo
from .scene_data import SceneData

# Lazy import for Isaac Lab dependencies
# Only loaded when primitive actions are actually used
_subtract_frame_transforms = None


@configclass
class PrimitiveActionCfg(ActionBaseCfg):
    """Base configuration for primitive actions.

    Attributes:
        agent_assets: Name(s) of articulated asset(s) acting as agents (e.g., robot manipulators). REQUIRED.
        timeout: Maximum time (in seconds) before timeout. Defaults to TIMEOUT_DEFAULT constant.
        action_space_info: Optional action space metadata for the controlled robot.
            Defines which indices of the action tensor control which DOFs.
            If None, actions will create full tensors (legacy behavior).
            If provided, actions will use masks for partial updates.

    """

    agent_assets: str | list[str] = MISSING
    timeout: float = TIMEOUT_DEFAULT
    action_space_info: ActionSpaceInfo | None = None

    def __post_init__(self):
        """Normalize agent_assets to list format."""
        if isinstance(self.agent_assets, str):
            self.agent_assets = [self.agent_assets]


def _get_subtract_frame_transforms():
    """Lazy import Isaac Lab's subtract_frame_transforms.

    Raises helpful error if Isaac Lab not available (e.g., on edge deployment).
    """
    global _subtract_frame_transforms
    if _subtract_frame_transforms is None:
        try:
            from isaaclab.utils.math import subtract_frame_transforms as sft

            _subtract_frame_transforms = sft
        except ImportError as e:
            raise ImportError(
                "Isaac Lab is required for primitive actions (Move, MoveToFrame, etc.). "
                "This is needed for IK solvers and frame transforms.\n\n"
                "If you're deploying to an edge device, consider:\n"
                "1. Using trained policies directly (not actions)\n"
                "2. Implementing custom lightweight actions\n"
                "3. Installing Isaac Lab if possible\n\n"
                f"Original error: {e}"
            ) from e
    return _subtract_frame_transforms


class PrimitiveAction(ActionBase):
    """Abstract base class for primitive state-machine actions."""

    def __init__(
        self,
        agent_assets: str | list[str],
        timeout: float,
        action_space_info: ActionSpaceInfo | None = None,
    ):
        """
        Args:
            agent_assets: Key(s) for articulated asset(s) acting as agents. Single string or list of strings. REQUIRED.
            timeout: Max time (in seconds) before the action times out.
            action_space_info: Optional action space metadata for mask creation.
                If provided, enables partial action updates via masks.

        Note:
            num_envs, device, and dt are set separately via set_execution_params() after construction.
            This is called automatically by StateMachine.
        """
        super().__init__()

        # Normalize to list format
        self.agent_assets = [agent_assets] if isinstance(agent_assets, str) else agent_assets
        self.timeout = timeout
        self.action_space_info = action_space_info

        # These will be set via set_execution_params()
        self.num_envs = None
        self.device = None
        self.dt = None
        self.time_elapsed = None
        self._env_success_mask = None
        self._env_failure_mask = None

    def set_execution_params(self, num_envs: int, device: str | torch.device, dt: float) -> None:
        """Set runtime execution parameters and initialize tensors.

        Called by StateMachine after action creation from config.

        Args:
            num_envs: Number of parallel environments.
            device: Device for tensor operations.
            dt: Simulation timestep in seconds.
        """
        self.num_envs = num_envs
        self.device = torch.device(device) if isinstance(device, str) else device
        self.dt = dt

        # Initialize tensors now that we have num_envs and device
        # Track elapsed time instead of steps
        self.time_elapsed = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._env_success_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._env_failure_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _convert_body_to_world_frame(
        self,
        object_pos_w: torch.Tensor,
        object_quat_w: torch.Tensor,
        body_pos: torch.Tensor,
        body_quat: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert body-frame (local) poses to world frame.

        This is used to transform manipulation frames (defined in object body frame)
        to world frame for targeting.

        Args:
            object_pos_w: (N, 3) object position in world frame.
            object_quat_w: (N, 4) object orientation quaternion (w,x,y,z) in world frame.
            body_pos: (N, 3) position in object body frame.
            body_quat: (N, 4) orientation quaternion (w,x,y,z) in object body frame, or None for identity.

        Returns:
            (world_positions, world_orientations): both shaped (N, 3/4).
        """
        # Ensure tensors on correct device
        object_pos_w = object_pos_w.to(self.device)
        object_quat_w = object_quat_w.to(self.device)
        body_pos = body_pos.to(self.device)

        if body_quat is None:
            body_quat = torch.zeros((body_pos.shape[0], 4), device=self.device)
            body_quat[:, 0] = 1.0  # identity quaternion (w=1, x=0, y=0, z=0)
        else:
            body_quat = body_quat.to(self.device)

        # Position: world_pos = object_pos_w + rotate(object_quat_w, body_pos)
        # We need to rotate body_pos by object_quat_w
        # Using quaternion rotation: v' = q * v * q^{-1}
        rotated_body_pos = quat_rotate(object_quat_w, body_pos)
        world_pos = object_pos_w + rotated_body_pos

        # Orientation: world_quat = object_quat_w * body_quat (quaternion multiplication)
        world_quat = quat_mul(object_quat_w, body_quat)

        return world_pos, world_quat

    def _convert_world_to_base_frame(
        self,
        scene_data: SceneData,
        asset_name: str,
        world_positions: torch.Tensor,
        world_orientations: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert world-frame poses to the asset's base frame.

        Args:
            scene_data: Scene state containing all articulations and rigid objects.
            asset_name: Name of the asset to use as reference frame.
            world_positions: (N, 3) world-frame positions.
            world_orientations: (N, 4) world-frame quaternions or None.

        Returns:
            (base_positions, base_orientations): both shaped (N, 3/4).
        """
        # Get asset pose from scene_data
        if asset_name in scene_data.articulations:
            asset_data = scene_data.articulations[asset_name]
            base_pos_w = asset_data.root_pos_w
            base_quat_w = asset_data.root_quat_w
        elif asset_name in scene_data.rigid_objects:
            asset_data = scene_data.rigid_objects[asset_name]
            base_pos_w = asset_data.pos_w
            base_quat_w = asset_data.quat_w
        else:
            raise ValueError(
                f"Asset '{asset_name}' not found in scene_data. "
                f"Available articulations: {list(scene_data.articulations.keys())}, "
                f"rigid_objects: {list(scene_data.rigid_objects.keys())}"
            )

        world_positions = world_positions.to(self.device)
        if world_orientations is None:
            world_orientations = torch.zeros((world_positions.shape[0], 4), device=self.device)
            world_orientations[:, 0] = 1.0  # identity (w, x, y, z)
        else:
            world_orientations = world_orientations.to(self.device)

        subtract_frame_transforms = _get_subtract_frame_transforms()
        base_positions, base_orientations = subtract_frame_transforms(
            base_pos_w.to(self.device),
            base_quat_w.to(self.device),
            world_positions,
            world_orientations,
        )
        return base_positions, base_orientations

    @abstractmethod
    def _compute_action_impl(self, scene_data: SceneData, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Implement per-step action computation.

        This method should ONLY compute the action command, not check for success/failure.
        Use _check_completion_impl to check for success/failure conditions.

        Args:
            scene_data: Complete scene state container.
            env_ids: Indices of active environments.

        Returns:
            (action_tensor, action_dim_mask):
                - action_tensor: Shape (num_envs, action_dim) - action values for all envs
                - action_dim_mask: Shape (action_dim,) - which dimensions this action controls
        """
        pass

    def _check_completion_impl(self, scene_data: SceneData, env_ids: torch.Tensor) -> None:
        """Check if action completed (success or custom failure) for active environments.

        Override this method to implement action-specific completion logic.
        Updates self._env_success_mask and self._env_failure_mask in-place.
        Timeout is handled automatically by parent class.

        Default implementation: Clears both masks (action runs until timeout).

        Args:
            scene_data: Complete scene state container.
            env_ids: Indices of active environments to check.

        Side effects:
            - Sets self._env_success_mask[env_ids] = True for succeeded environments
            - Sets self._env_failure_mask[env_ids] = True for custom failures
            - Both masks are pre-allocated in __init__ for efficiency
        """
        # Default: no completion checking, action runs until timeout
        # Clear masks for active environments (other envs keep their old values)
        self._env_success_mask[env_ids] = False
        self._env_failure_mask[env_ids] = False

    def compute_action(
        self, scene_data: SceneData, env_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Advance the action one step for the given environments.

        Args:
            scene_data: Complete scene state container.
            env_ids: Indices of active environments.

        Returns:
            (action_tensor, action_dim_mask, env_success_mask, env_failure_mask):
                - action_tensor: Shape (num_envs, action_dim) - action values for all envs
                - action_dim_mask: Shape (action_dim,) - which dimensions this action controls
                - env_success_mask: Shape (num_envs,) - which environments completed successfully
                - env_failure_mask: Shape (num_envs,) - which environments failed (timeout or custom)
        """
        # Increment elapsed time by dt (one simulation step)
        self.time_elapsed[env_ids] += self.dt

        # Check completion BEFORE computing action (more logical flow)
        self._check_completion_impl(scene_data, env_ids)

        # Check timeout (automatic failure mode)
        env_timeout_mask = self.time_elapsed >= self.timeout

        # Combine timeout with custom failures
        env_failure_mask = env_timeout_mask | self._env_failure_mask

        # Compute action command (still needed even if completed, for state machine)
        action_tensor, action_dim_mask = self._compute_action_impl(scene_data, env_ids)

        # Return pre-computed success mask from _check_completion_impl
        return action_tensor, action_dim_mask, self._env_success_mask, env_failure_mask

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset internal time counters and action-specific state.

        Args:
            env_ids: Indices of environments to reset. If None, resets all environments.
        """
        if env_ids is None:
            # Reset all environments
            self.time_elapsed.zero_()
            self._env_success_mask.zero_()
            self._env_failure_mask.zero_()
        else:
            # Reset specific environments
            self.time_elapsed[env_ids] = 0.0
            self._env_success_mask[env_ids] = False
            self._env_failure_mask[env_ids] = False

        # Call subclass-specific reset hook
        self._reset_impl(env_ids)

    def _reset_impl(self, env_ids: torch.Tensor | None = None) -> None:
        """Subclass hook for resetting action-specific state.

        Override this in subclasses to reset action-specific state
        (e.g., cached targets, initialization flags).

        Args:
            env_ids: Indices of environments being reset, or None for all.
        """
        pass  # Default: no action-specific state to reset

    def _create_action_mask(self, control_type: str) -> torch.Tensor:
        """Create boolean mask indicating which action dimensions this action controls.

        Args:
            control_type: Type of control this action performs:
                - "position": Controls position dimensions only
                - "orientation": Controls orientation dimensions only
                - "gripper": Controls gripper dimensions only
                - "position_orientation": Controls position + orientation
                - "full": Controls all dimensions

        Returns:
            Boolean tensor of shape (action_dim,) where True indicates controlled dimensions.

        Raises:
            ValueError: If action_space_info is None or control_type is unknown.
        """
        if self.action_space_info is None:
            raise ValueError(
                "action_space_info is required to create action masks. "
                "Pass action_space_info parameter when creating the action."
            )

        mask = torch.zeros(self.action_space_info.total_dim, dtype=torch.bool, device=self.device)

        if control_type == "position":
            if self.action_space_info.position_indices is not None:
                mask[list(self.action_space_info.position_indices)] = True
        elif control_type == "orientation":
            if self.action_space_info.orientation_indices is not None:
                mask[list(self.action_space_info.orientation_indices)] = True
        elif control_type == "gripper":
            if self.action_space_info.gripper_indices is not None:
                mask[list(self.action_space_info.gripper_indices)] = True
        elif control_type == "position_orientation":
            if self.action_space_info.position_indices is not None:
                mask[list(self.action_space_info.position_indices)] = True
            if self.action_space_info.orientation_indices is not None:
                mask[list(self.action_space_info.orientation_indices)] = True
        elif control_type == "full":
            mask[:] = True
        else:
            raise ValueError(f"Unknown control_type: {control_type}")

        return mask

    @classmethod
    def from_cfg(cls, cfg: PrimitiveActionCfg):
        """Create action instance from configuration.

        Note: num_envs and device will be set later via set_execution_params() by StateMachine.

        Args:
            cfg: Action configuration object.

        Returns:
            Action instance initialized with config parameters.
        """
        return cls(
            agent_assets=cfg.agent_assets,
            timeout=cfg.timeout,
            action_space_info=getattr(cfg, "action_space_info", None),
        )
