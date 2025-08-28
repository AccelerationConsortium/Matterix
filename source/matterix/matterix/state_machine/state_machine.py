# Copyright (c) 2022-2025, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from typing import Tuple

from .compositional_action import CompositionalAction
from .workflow_env import WorkflowEnv


class StateMachine:
    """Simple sequential state machine orchestrating a list of actions over parallel environments."""

    def __init__(self, env: WorkflowEnv, device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        """
        Args:
            env: Environment proxy exposing robot/object state for parallel envs.
            device: Torch device used for internal tensors.
        """
        print(env)
        self.env = env
        self.num_envs = env.num_envs
        self.device = device

        self.current_action_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.actions = []
        self.all_done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.obs_history = None
        self.frame_history = []

    def reset(self):
        """Reset per-episode bookkeeping and all registered actions."""
        self.current_action_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.action_sequence_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.action_sequence_failure = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.combined_action = torch.zeros((self.num_envs, 8), device=self.device)
        for action in self.actions:
            action.reset()

    def set_action_sequence(self, actions):
        """Register a new action sequence.

        Compositional actions are expanded into their constituent actions.
        All actions are reset.

        Args:
            actions: Iterable of Action or CompositionalAction instances.
        """
        self.actions = []
        for action in actions:
            action.reset()
            if isinstance(action, CompositionalAction):
                action.initialize(self.env)
                self.actions += action.actions_list
            else:
                self.actions.append(action)
        self.obs_history = {}
        self.frame_history = []

    def execute_action_sequence(self, actions) -> tuple[torch.Tensor, str]:
        """Run a full sequence until success or failure for all environments.

        Args:
            actions: Iterable of Action or CompositionalAction instances.

        Returns:
            Tuple:
                - success mask (num_envs,) bool
                - frame history (opaque log/list as produced during stepping)
        """
        self.set_action_sequence(actions)
        while not (self.action_sequence_success | self.action_sequence_failure).all():
            self.step()
        return self.action_sequence_success, self.frame_history

    def step(self):
        """Advance exactly one step for all environments.

        Executes the current action for each active environment, updates per-env
        success/failure masks, and advances to the next action when complete.

        Returns:
            torch.Tensor: The combined action tensor sent to the environment
                          with shape (num_envs, action_dim).
        """
        active_mask = ~(self.action_sequence_success | self.action_sequence_failure)
        unique_action_indices = torch.unique(self.current_action_idx[active_mask])

        combined_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        for action_idx in unique_action_indices:
            action_mask = (self.current_action_idx == action_idx) & active_mask
            action_env_ids = torch.nonzero(action_mask).squeeze(-1)
            action = self.actions[action_idx]

            action_values, action_success, action_failure = action.compute_action(self.env, action_env_ids)
            action_values = action_values.to(self.device)
            action_success = action_success.to(self.device)
            action_failure = action_failure.to(self.device)

            self.combined_action[action_env_ids] = action_values
            combined_success[action_env_ids] = action_success
            self.action_sequence_failure = self.action_sequence_failure | action_failure

        self.current_action_idx[combined_success] += 1
        self.action_sequence_success = self.action_sequence_success | (self.current_action_idx >= len(self.actions))

        return self.combined_action
