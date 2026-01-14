# Copyright (c) 2022-2026, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MoveToFrame action - move end-effector to an object's named frame."""

from __future__ import annotations

import torch
from dataclasses import MISSING
from typing import ClassVar

from .._compat import configclass
from ..math_utils import quat_mul, quat_rotate
from ..robot_action_spaces import ActionSpaceInfo
from ..scene_data import SceneData
from .move_to_pose import MoveToPose, MoveToPoseCfg


@configclass
class MoveToFrameCfg(MoveToPoseCfg):
    """Configuration for MoveToFrame action (move to object frame).

    Inherits from MoveToPoseCfg and adds object/frame specific fields.

    Attributes:
        object: Name of the object with the target frame. REQUIRED.
        frame: Name of the frame to move to (e.g., "grasp", "pre_grasp"). REQUIRED.
    """

    object: str = MISSING
    frame: str = MISSING


class MoveToFrame(MoveToPose):
    """Move the end-effector to an object's named frame (offset) in world coordinates.

    Inherits from MoveToPose. On first call, looks up the frame pose from the object
    and uses it as the target. Subsequent calls use the cached target.
    """

    cfg_type: ClassVar[type] = MoveToFrameCfg

    def __init__(
        self,
        object: str,
        frame: str,
        agent_assets: str | list[str],
        timeout: float,
        position_threshold: float,
        orientation_threshold: float,
        action_space_info: ActionSpaceInfo | None = None,
    ):
        """
        Args:
            object: Object name in scene_data.rigid_objects.
            frame: Named frame key under the object.
            agent_assets: Name(s) of articulated asset(s) acting as agents.
            timeout: Max time (in seconds) before timeout.
            position_threshold: Distance threshold for success (meters).
            orientation_threshold: Orientation threshold for success (radians).
            action_space_info: Optional action space metadata for mask creation.
        """
        # Initialize parent with None targets (will be set on first call)
        super().__init__(
            agent_assets=agent_assets,
            target_positions_w=None,
            target_orientations_w=None,
            timeout=timeout,
            position_threshold=position_threshold,
            orientation_threshold=orientation_threshold,
            action_space_info=action_space_info,
        )

        # Store frame lookup info
        self.object = object
        self.frame = frame
        self._frame_initialized = False

    def _compute_action_impl(self, scene_data: SceneData, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute move-to-frame action for controlled asset.

        On first call, looks up the frame pose from the object and caches it as the target.
        Subsequent calls use the cached target pose.

        Args:
            scene_data: Complete scene state container.
            env_ids: Indices of active environments.

        Returns:
            (action_tensor, action_dim_mask):
                - action_tensor: Shape (num_envs, action_dim) - action values for all envs
                - action_dim_mask: Shape (action_dim,) - which dimensions this action controls
        """
        # On first call, look up frame pose and cache it as target
        if not self._frame_initialized:
            # Get target object data
            if self.object not in scene_data.rigid_objects:
                raise ValueError(
                    f"Object '{self.object}' not found in scene_data.rigid_objects. "
                    f"Available: {list(scene_data.rigid_objects.keys())}"
                )

            obj_data = scene_data.rigid_objects[self.object]

            # Get target frame pose (in object body frame)
            if obj_data.frames is None or self.frame not in obj_data.frames:
                available = list(obj_data.frames.keys()) if obj_data.frames else []
                raise ValueError(
                    f"Frame '{self.frame}' not found for object '{self.object}'. Available frames: {available}"
                )

            frame_pose = obj_data.frames[self.frame]

            # Frames are already in world frame (transformed by Isaac Lab's FrameTransformer)
            grasp_pos_w = frame_pose.position.to(self.device)
            grasp_quat_w = frame_pose.orientation.to(self.device) if frame_pose.orientation is not None else None

            # Apply robot-specific grasp-to-EE offset: ^W T_ee = ^W T_g · ^g T_ee
            if self.action_space_info and self.action_space_info.grasp_to_ee_offset and grasp_quat_w is not None:
                offset_pos, offset_quat = self.action_space_info.grasp_to_ee_offset
                offset_pos_t = (
                    torch.tensor(offset_pos, device=self.device, dtype=torch.float32)
                    .unsqueeze(0)
                    .expand(self.num_envs, -1)
                )
                offset_quat_t = (
                    torch.tensor(offset_quat, device=self.device, dtype=torch.float32)
                    .unsqueeze(0)
                    .expand(self.num_envs, -1)
                )

                self.target_positions_w = grasp_pos_w + quat_rotate(grasp_quat_w, offset_pos_t)
                self.target_orientations_w = quat_mul(grasp_quat_w, offset_quat_t)
            else:
                self.target_positions_w = grasp_pos_w
                self.target_orientations_w = grasp_quat_w

            self._frame_initialized = True

        # Delegate to parent's implementation
        return super()._compute_action_impl(scene_data, env_ids)

    def _reset_impl(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset frame initialization flag when environments are reset."""
        # Frame needs to be re-queried after reset
        self._frame_initialized = False

    @classmethod
    def from_cfg(cls, cfg: MoveToFrameCfg):
        """Create MoveToFrame action from configuration."""
        return cls(
            object=cfg.object,
            frame=cfg.frame,
            agent_assets=cfg.agent_assets,
            timeout=cfg.timeout,
            position_threshold=cfg.position_threshold,
            orientation_threshold=cfg.orientation_threshold,
            action_space_info=cfg.action_space_info,
        )
