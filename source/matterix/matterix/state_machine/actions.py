# Copyright (c) 2022-2025, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from abc import ABC, abstractmethod
from typing import Tuple

from isaaclab.utils.math import subtract_frame_transforms


class Action(ABC):
    def __init__(
        self,
        asset="robot",
        num_envs: int = 1,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        max_duration: int = 100,
    ):
        self.asset = asset
        self.num_envs = num_envs
        self.device = device
        self.max_duration = max_duration
        self.steps_taken = torch.zeros(num_envs, dtype=torch.int32, device=device)

    def _convert_world_to_base_frame(
        self, env, env_ids: torch.Tensor, world_positions: torch.Tensor, world_orientations: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert world frame poses to robot base frame poses.

        Args:
            env: The environment to get transforms from
            env_ids: Tensor of environment indices
            world_positions: Positions in world frame, shape (len(env_ids), 3)
            world_orientations: Orientations in world frame, shape (len(env_ids), 4). If None, returns zero quaternions.

        Returns:
            Tuple of (base_positions, base_orientations) where:
            - base_positions: Positions in robot base frame, shape (len(env_ids), 3)
            - base_orientations: Orientations in robot base frame, shape (len(env_ids), 4)
        """
        env = env.unwrapped
        root_pose_w = env.scene[self.asset].data.root_state_w[env_ids, :7]

        # Use identity quaternions if orientations not provided
        if world_orientations is None:
            world_orientations = torch.zeros((len(env_ids), 4), device=self.device)
            world_orientations[:, 0] = 1.0  # Identity quaternion [1, 0, 0, 0] (w, x, y, z)

        # Convert world poses to base frame using proper transform
        base_positions, base_orientations = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], world_positions, world_orientations
        )

        return base_positions, base_orientations

    @abstractmethod
    def _compute_action_impl(self, env, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the action values for environments that are not done.

        Args:
            env: The environment to get state from
            env_ids: Boolean mask indicating which environments are active

        Returns:
            Tuple of (action_values, done_mask) where:
            - action_values is the action tensor to apply
            - done_mask indicates which environments completed this action
        """
        pass

    def compute_action(self, env, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Update steps and check timeout
        # TODO: Fail and finish at the same time should probably be a success (we already did the work, might as well mark it as success)
        self.steps_taken[env_ids] += 1
        timeout_failure = self.steps_taken >= self.max_duration

        action, success = self._compute_action_impl(env, env_ids)
        return action, success, timeout_failure

    def reset(self):
        """Reset the action's internal state"""
        self.steps_taken = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)


class Move(Action):
    def __init__(
        self,
        target_positions_w: torch.Tensor,
        asset="robot",
        num_envs: int = 1,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        max_duration: int = 100,
    ):
        """
        Args:
            num_envs: Number of parallel environments
            device: Device to use for tensors
            target_positions: Tensor of shape (num_envs, 3) containing target positions
        """
        super().__init__(asset, num_envs, device, max_duration)

        self.target_positions_w = target_positions_w
        self.position_threshold = 0.01  # Distance threshold to consider target reached
        self.gripper_threshold = 0.001

    def _compute_action_impl(self, env, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Convert target positions from world frame to robot base frame (why do this computation each step?)
        target_positions_b, _ = self._convert_world_to_base_frame(env, env_ids, self.target_positions_w[env_ids])

        # Create action tensor
        action = torch.zeros((env_ids.shape[0], 8), device=self.device)
        action[:, 0:3] = target_positions_b

        # Keep orientation fixed (pointing down)
        action[:, 3] = 0.0
        action[:, 4] = 0.7071
        action[:, 5] = 0.7071
        action[:, 6] = 0.0

        # We need to send some action to the gripper
        # Our goal is not to have it do anything
        # If it is fully open, we keep sending the open command
        # In any other case if it is static it will the fully closed or holding onto something
        # (as it can only be controlled by binary action)
        # Therefore, we keep sending the close command
        # We are doing this to avoid any memory in the system (remembering the last action)
        gripper_pos = env.scene[self.asset].data.joint_pos[env_ids, -1]
        open_gripper = gripper_pos >= 0.04 - self.gripper_threshold
        close_gripper = ~open_gripper
        action[open_gripper, 7] = 1.0
        action[close_gripper, 7] = -1.0

        # we assume the robot has a sensor for the ee_frame
        ee_frame_sensor = env.unwrapped.scene["ee_frame"]
        current_pos = ee_frame_sensor.data.target_pos_w[env_ids, 0, :].clone()

        # TODO: Check if both of these are in full world and not env world frame
        distance = torch.norm(self.target_positions_w[env_ids] - current_pos, dim=1)
        new_done = distance < self.position_threshold

        return action, new_done


class MoveToFrame(Action):
    def __init__(
        self,
        object: str,
        frame: str,
        asset="robot",
        num_envs: int = 1,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        max_duration: int = 100,
    ):
        """
        Args:
            num_envs: Number of parallel environments
            device: Device to use for tensors
            target_positions: Tensor of shape (num_envs, 3) containing target positions
        """
        super().__init__(asset, num_envs, device, max_duration)

        self.object = object
        self.frame = frame
        self.position_threshold = 0.01  # Distance threshold to consider target reached
        self.gripper_threshold = 0.001

    def _compute_action_impl(self, env, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Convert target positions from world frame to robot base frame
        target_positions_w = env.unwrapped.scene[self.object].data.root_pos_w + torch.tensor(
            env.unwrapped.scene[self.object].cfg.frames[self.frame], device=self.device
        )
        target_positions_b, _ = self._convert_world_to_base_frame(env, env_ids, target_positions_w[env_ids])

        # Create action tensor
        action = torch.zeros((env_ids.shape[0], 8), device=self.device)
        action[:, 0:3] = target_positions_b

        # Keep orientation fixed (pointing down)
        action[:, 3] = 0.0
        action[:, 4] = 0.7071
        action[:, 5] = 0.7071
        action[:, 6] = 0.0

        # We need to send some action to the gripper
        # Our goal is not to have it do anything
        # If it is fully open, we keep sending the open command
        # In any other case if it is static it will the fully closed or holding onto something
        # (as it can only be controlled by binary action)
        # Therefore, we keep sending the close command
        # We are doing this to avoid any memory in the system (remembering the last action)
        gripper_pos = env.scene[self.asset].data.joint_pos[env_ids, -1]
        open_gripper = gripper_pos >= 0.04 - self.gripper_threshold
        close_gripper = ~open_gripper
        action[open_gripper, 7] = 1.0
        action[close_gripper, 7] = -1.0

        # we assume the robot has a sensor for the ee_frame
        ee_frame_sensor = env.unwrapped.scene["ee_frame"]
        current_pos = ee_frame_sensor.data.target_pos_w[env_ids, 0, :].clone()

        # TODO: Check if both of these are in full world and not env world frame
        distance = torch.norm(target_positions_w[env_ids] - current_pos, dim=1)
        new_done = distance < self.position_threshold

        return action, new_done


class MoveRelative(Action):
    def __init__(
        self,
        offset,
        asset="robot",
        num_envs: int = 1,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        max_duration: int = 100,
    ):
        """
        Args:
            num_envs: Number of parallel environments
            device: Device to use for tensors
            target_positions: Tensor of shape (num_envs, 3) containing target positions
        """
        super().__init__(asset, num_envs, device, max_duration)

        self.offset = offset
        self.target_positions_w = None
        self.position_threshold = 0.01  # Distance threshold to consider target reached
        self.gripper_threshold = 0.001

    def _compute_action_impl(self, env, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Convert target positions from world frame to robot base frame
        if self.target_positions_w is None:
            self.target_positions_w = env.unwrapped.scene["ee_frame"].data.target_pos_w[:, 0, :].clone() + torch.tensor(
                self.offset, device=self.device
            )

        target_positions_b, _ = self._convert_world_to_base_frame(env, env_ids, self.target_positions_w[env_ids])

        # Create action tensor
        action = torch.zeros((env_ids.shape[0], 8), device=self.device)
        action[:, 0:3] = target_positions_b

        # Keep orientation fixed (pointing down)
        action[:, 3] = 0.0
        action[:, 4] = 0.7071
        action[:, 5] = 0.7071
        action[:, 6] = 0.0

        # We need to send some action to the gripper
        # Our goal is not to have it do anything
        # If it is fully open, we keep sending the open command
        # In any other case if it is static it will the fully closed or holding onto something
        # (as it can only be controlled by binary action)
        # Therefore, we keep sending the close command
        # We are doing this to avoid any memory in the system (remembering the last action)

        gripper_pos = env.scene[self.asset].data.joint_pos[env_ids, -1]
        open_gripper = gripper_pos >= 0.04 - self.gripper_threshold
        close_gripper = ~open_gripper
        action[open_gripper, 7] = 1.0
        action[close_gripper, 7] = -1.0

        # we assume the robot has a sensor for the ee_frame
        ee_frame_sensor = env.unwrapped.scene["ee_frame"]
        current_pos = ee_frame_sensor.data.target_pos_w[env_ids, 0, :].clone()

        # TODO: Check if both of these are in full world and not env world frame
        distance = torch.norm(self.target_positions_w[env_ids] - current_pos, dim=1)
        new_done = distance < self.position_threshold

        return action, new_done


class GripperAction(Action):
    def __init__(
        self,
        asset="robot",
        num_envs: int = 1,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        target_value: float = 0,
        duration: int = 20,
    ):
        super().__init__(asset, num_envs, device)
        self.target_value = target_value
        self.duration = duration

    def _compute_action_impl(self, env, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Create action tensor
        action = torch.zeros((env_ids.shape[0], 8), device=self.device)

        # Get current end effector position in world frame and convert to base frame
        env = env.unwrapped
        ee_pos_w = env.scene["ee_frame"].data.target_pos_w[env_ids, 0, :]
        ee_pos_b, _ = self._convert_world_to_base_frame(env, env_ids, ee_pos_w)

        action[:, 0:3] = ee_pos_b

        # Keep orientation fixed (pointing down)
        action[:, 3] = 0.0
        action[:, 4] = 0.7071
        action[:, 5] = 0.7071
        action[:, 6] = 0.0

        action[:, 7] = self.target_value

        new_done = self.steps_taken[env_ids] >= self.duration

        return action, new_done


class OpenGripper(GripperAction):
    def __init__(
        self,
        asset="robot",
        num_envs: int = 1,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        super().__init__(asset, num_envs, device, target_value=1.0)


class CloseGripper(GripperAction):
    def __init__(
        self,
        asset="robot",
        num_envs: int = 1,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        super().__init__(asset, num_envs, device, target_value=-1.0)
