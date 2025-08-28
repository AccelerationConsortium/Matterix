# Copyright (c) 2022-2025, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from abc import ABC, abstractmethod
from typing import Tuple

from isaaclab.utils.math import subtract_frame_transforms
from .workflow_env import WorkflowEnv


class Action(ABC):
    """Abstract base class for state-machine actions."""

    def __init__(
        self,
        asset: str = "robot",
        num_envs: int = 1,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        max_duration: int = 100,
    ):
        """
        Args:
            asset: Key for the controlled asset in the workflow env.
            num_envs: Number of parallel environments.
            device: Torch device for tensors.
            max_duration: Max steps before the action times out.
        """
        self.asset = asset
        self.num_envs = num_envs
        self.device = device
        self.max_duration = max_duration
        self.steps_taken = torch.zeros(num_envs, dtype=torch.int32, device=device)

    def _convert_world_to_base_frame(
        self,
        env: WorkflowEnv,
        env_ids: torch.Tensor,
        world_positions: torch.Tensor,
        world_orientations: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert world-frame poses to the asset's base frame.

        Args:
            env: Workflow environment providing transforms.
            env_ids: Indices of active environments.
            world_positions: (N, 3) world-frame positions.
            world_orientations: (N, 4) world-frame quaternions or None.

        Returns:
            (base_positions, base_orientations): both shaped (N, 3/4).
        """
        if self.asset in env.objects:
            pose = env.objects[self.asset].pose
        else:
            pose = env.robots[self.asset].pose

        world_positions = world_positions.to(self.device)
        if world_orientations is None:
            world_orientations = torch.zeros((len(env_ids), 4), device=self.device)
            world_orientations[:, 0] = 1.0  # identity (w, x, y, z)
        else:
            world_orientations = world_orientations.to(self.device)

        base_positions, base_orientations = subtract_frame_transforms(
            pose.position.to(self.device), pose.orientation.to(self.device), world_positions, world_orientations
        )
        return base_positions, base_orientations

    @abstractmethod
    def _compute_action_impl(self, env: WorkflowEnv, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Implement per-step action computation.

        Args:
            env: Workflow environment.
            env_ids: Active environment indices.

        Returns:
            (action, done_mask): action shaped (N, A), done_mask shaped (N,).
        """
        pass

    def compute_action(self, env: WorkflowEnv, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Advance the action one step for the given environments.

        Returns:
            (action, success_mask, timeout_mask)
        """
        self.steps_taken[env_ids] += 1
        timeout_failure = self.steps_taken >= self.max_duration
        action, success = self._compute_action_impl(env, env_ids)
        return action, success, timeout_failure

    def reset(self) -> None:
        """Reset internal step counters."""
        self.steps_taken = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)


class Move(Action):
    """Move the end-effector to target world positions with fixed orientation and binary gripper command."""

    def __init__(
        self,
        target_positions_w: torch.Tensor,
        asset: str = "robot",
        num_envs: int = 1,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        max_duration: int = 100,
    ):
        """
        Args:
            target_positions_w: (num_envs, 3) world-frame targets.
        """
        super().__init__(asset, num_envs, device, max_duration)
        self.target_positions_w = target_positions_w.to(device)
        self.position_threshold = 0.01
        self.gripper_threshold = 0.001

    def _compute_action_impl(self, env: WorkflowEnv, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        target_positions_b, _ = self._convert_world_to_base_frame(env, env_ids, self.target_positions_w[env_ids])

        action = torch.zeros((env_ids.shape[0], 8), device=self.device)
        action[:, 0:3] = target_positions_b
        action[:, 3] = 0.0
        action[:, 4] = 0.7071
        action[:, 5] = 0.7071
        action[:, 6] = 0.0

        gripper_pos = env.robots[self.asset].joint_positions.to(self.device)[env_ids, -1]
        open_gripper = gripper_pos >= 0.04 - self.gripper_threshold
        close_gripper = ~open_gripper
        action[open_gripper, 7] = 1.0
        action[close_gripper, 7] = -1.0

        current_pos = env.robots[self.asset].ee_position[env_ids].to(self.device)  # type: ignore
        distance = torch.norm(self.target_positions_w[env_ids] - current_pos, dim=1)
        new_done = distance < self.position_threshold
        return action, new_done


class MoveToFrame(Action):
    """Move the end-effector to an object’s named frame (offset) in world coordinates."""

    def __init__(
        self,
        object: str,
        frame: str,
        asset: str = "robot",
        num_envs: int = 1,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        max_duration: int = 100,
    ):
        """
        Args:
            object: Object key in the workflow env.
            frame: Named frame key under the object.
        """
        super().__init__(asset, num_envs, device, max_duration)
        self.object = object
        self.frame = frame
        self.position_threshold = 0.01
        self.gripper_threshold = 0.001

    def _compute_action_impl(self, env: WorkflowEnv, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        target_positions_w = env.objects[self.object].pose.position.to(self.device) + torch.as_tensor(
            env.objects[self.object].frames[self.frame], device=self.device
        ).unsqueeze(0)
        target_positions_b, _ = self._convert_world_to_base_frame(env, env_ids, target_positions_w[env_ids])

        action = torch.zeros((env_ids.shape[0], 8), device=self.device)
        action[:, 0:3] = target_positions_b
        action[:, 3] = 0.0
        action[:, 4] = 0.7071
        action[:, 5] = 0.7071
        action[:, 6] = 0.0

        gripper_pos = env.robots[self.asset].joint_positions.to(self.device)[env_ids, -1]
        open_gripper = gripper_pos >= 0.04 - self.gripper_threshold
        close_gripper = ~open_gripper
        action[open_gripper, 7] = 1.0
        action[close_gripper, 7] = -1.0

        current_pos = env.robots[self.asset].ee_position[env_ids].to(self.device)  # type: ignore
        distance = torch.norm(target_positions_w[env_ids] - current_pos, dim=1)
        new_done = distance < self.position_threshold
        return action, new_done


class MoveRelative(Action):
    """Move the end-effector by a constant world-frame offset relative to the current EE position."""

    def __init__(
        self,
        offset,
        asset: str = "robot",
        num_envs: int = 1,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        max_duration: int = 100,
    ):
        """
        Args:
            offset: (3,) translation in world frame applied once to form the target.
        """
        super().__init__(asset, num_envs, device, max_duration)
        self.offset = offset
        self.target_positions_w = None
        self.position_threshold = 0.01
        self.gripper_threshold = 0.001

    def _compute_action_impl(self, env: WorkflowEnv, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.target_positions_w is None:
            self.target_positions_w = env.robots[self.asset].ee_position[env_ids].to(self.device) + torch.tensor(  # type: ignore
                self.offset, device=self.device
            )

        target_positions_b, _ = self._convert_world_to_base_frame(env, env_ids, self.target_positions_w[env_ids])

        action = torch.zeros((env_ids.shape[0], 8), device=self.device)
        action[:, 0:3] = target_positions_b
        action[:, 3] = 0.0
        action[:, 4] = 0.7071
        action[:, 5] = 0.7071
        action[:, 6] = 0.0

        gripper_pos = env.robots[self.asset].joint_positions.to(self.device)[env_ids, -1]
        open_gripper = gripper_pos >= 0.04 - self.gripper_threshold
        close_gripper = ~open_gripper
        action[open_gripper, 7] = 1.0
        action[close_gripper, 7] = -1.0

        current_pos = env.robots[self.asset].ee_position[env_ids].to(self.device)  # type: ignore
        distance = torch.norm(self.target_positions_w[env_ids] - current_pos, dim=1)
        new_done = distance < self.position_threshold
        return action, new_done


class GripperAction(Action):
    """Hold current EE pose and apply a binary gripper command for a fixed duration."""

    def __init__(
        self,
        asset: str = "robot",
        num_envs: int = 1,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        target_value: float = 0.0,
        duration: int = 20,
    ):
        """
        Args:
            target_value: Gripper command value.
            duration: Steps to maintain the command.
        """
        super().__init__(asset, num_envs, device)
        self.target_value = target_value
        self.duration = duration

    def _compute_action_impl(self, env: WorkflowEnv, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        action = torch.zeros((env_ids.shape[0], 8), device=self.device)

        ee_pos_w = env.robots[self.asset].ee_position[env_ids].to(self.device)  # type: ignore
        ee_pos_b, _ = self._convert_world_to_base_frame(env, env_ids, ee_pos_w)

        action[:, 0:3] = ee_pos_b
        action[:, 3] = 0.0
        action[:, 4] = 0.7071
        action[:, 5] = 0.7071
        action[:, 6] = 0.0
        action[:, 7] = self.target_value

        new_done = self.steps_taken[env_ids] >= self.duration
        return action, new_done


class OpenGripper(GripperAction):
    """Open the gripper while holding the current pose."""

    def __init__(
        self,
        asset: str = "robot",
        num_envs: int = 1,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        super().__init__(asset, num_envs, device, target_value=1.0)


class CloseGripper(GripperAction):
    """Close the gripper while holding the current pose."""

    def __init__(
        self,
        asset: str = "robot",
        num_envs: int = 1,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        super().__init__(asset, num_envs, device, target_value=-1.0)
