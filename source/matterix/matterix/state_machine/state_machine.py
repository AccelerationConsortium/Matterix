# Copyright (c) 2022-2025, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from typing import Tuple

from .compositional_action import CompositionalAction


class StateMachine:
    def __init__(self, env):
        """
        Args:
            env: The underlying environment that supports parallel execution
        """
        self.env = env
        self.num_envs = env.num_envs
        self.device = env.unwrapped.device

        # State machine tracking
        self.current_action_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.actions = []
        self.all_done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Observation history tracking
        self.obs_history = None  # Will be initialized in set_action_sequence
        self.frame_history = []

    def reset(self):
        """Reset the environment and action state"""
        # TODO: Add independent resets (would need to update actions with specific new positions usually)
        #       Maybe action can have a function attached, so that it calculates in real time (like obs terms)
        self.obs_dict = self.env.reset()[0]
        self.current_action_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.action_sequence_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.action_sequence_failure = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.combined_action = torch.zeros((self.num_envs, 8), device=self.device)

        # Reset all actions
        for action in self.actions:
            action.reset()

        return self.obs_dict

    def set_action_sequence(self, actions):
        """
        Set the sequence of actions to execute.

        Args:
            actions: List of Action objects to execute in sequence
        """
        for _ in range(15):  # let physics settle
            self.env.sim.step(render=False)
        self.actions = []
        # Reset all actions
        for action in actions:
            action.reset()
            if isinstance(action, CompositionalAction):
                action.initialize(self.env)
                self.actions += action.actions_list
            else:
                self.actions.append(action)
        # Initialize observation history tensors
        self.obs_history = {}
        for k, v in self.obs_dict.items():
            # Shape: (num_actions + 1, num_envs, K) - +1 for initial state
            self.obs_history[k] = torch.zeros((len(actions) + 1, self.num_envs, v.shape[1]), device=self.device)
            self.obs_history[k][0] = v  # Store initial observations

        self.frame_history = []

    def execute_action_sequence(self, actions) -> tuple[torch.Tensor, str]:
        """Execute a sequence of actions and return success status and observation summary.

        Args:
            actions: List of Action objects to execute in sequence

        Returns:
            Tuple containing:
            - torch.Tensor: Boolean tensor indicating which environments succeeded
            - str: Summary of observation changes during execution
        """
        self.set_action_sequence(actions)
        while not (self.action_sequence_success | self.action_sequence_failure).all():
            self.step()

        return self.action_sequence_success, self.frame_history

    def step(self):
        """
        Execute one step of the current action for all environments.
        Environments that complete their current action will move to the next action.

        Returns:
            observation: The current observation
            done: Boolean tensor indicating which environments have completed all actions
        """
        # Create mask for environments that are still active
        active_mask = ~(self.action_sequence_success | self.action_sequence_failure)

        # TODO: Still running loop now, even if everything is done
        # if not active_mask.any():
        #     return None, self.all_done

        # Get all unique action indices that are currently active
        unique_action_indices = torch.unique(self.current_action_idx[active_mask])

        # Initialize action tensor
        # TODO: Slightly inelegant, for envs that are done, we want to keep sending the last action to stay in place
        # combined_action = torch.zeros((self.num_envs, 8), device=self.device)
        combined_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Compute actions for each unique action type
        for action_idx in unique_action_indices:
            # Get mask for environments executing this action
            action_mask = (self.current_action_idx == action_idx) & active_mask
            # Convert boolean mask to indices
            action_env_ids = torch.nonzero(action_mask).squeeze(-1)
            # Get the action and compute its values
            action = self.actions[action_idx]
            action_values, action_success, action_failure = action.compute_action(self.env, action_env_ids)

            # Update combined action and done masks
            self.combined_action[action_env_ids] = action_values
            combined_success[action_env_ids] = action_success

            # Store observations for environments that completed this action

            self.action_sequence_failure = self.action_sequence_failure | action_failure

        # Step the environment ONCE with the combined actions
        self.obs_dict = self.env.step(self.combined_action)[0]

        # Update action indices only for environments that completed their current action
        self.current_action_idx[combined_success] += 1

        # Mark environments as done if they've completed all actions
        self.action_sequence_success = self.action_sequence_success | (self.current_action_idx >= len(self.actions))

        return self.obs_dict, self.action_sequence_success

    def close(self):
        self.env.close()

    @property
    def unwrapped(self):
        return self.env.unwrapped
