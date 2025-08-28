# Copyright (c) 2022-2025, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

from .actions import * # noqa: F403
from .workflow_env import WorkflowEnv

class CompositionalAction:

    def __init__(
        self,
        asset="robot",
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        max_duration: int = 100,
    ):
        self.actions_list = []
        self.asset = asset
        self.device = (device,)
        self.max_duration = max_duration

    def initialize(self, env):
        raise NotImplementedError


class PickObject(CompositionalAction):

    def __init__(
        self,
        object,
        asset="robot",
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        max_duration: int = 100,
    ):
        super().__init__(asset, device, max_duration)
        self.object = object

    def initialize(self, env: WorkflowEnv):
        self.actions_list = [
            MoveToFrame(
                object=self.object,
                frame="pre_grasp",
                asset=self.asset,
                num_envs=env.num_envs,
                device=env.device,
                max_duration=self.max_duration,
            ),
            OpenGripper(asset=self.asset, num_envs=env.num_envs, device=env.device),
            MoveToFrame(
                object=self.object,
                frame="grasp",
                asset=self.asset,
                num_envs=env.num_envs,
                device=env.device,
                max_duration=self.max_duration,
            ),
            CloseGripper(asset=self.asset, num_envs=env.num_envs, device=env.device),
            MoveRelative(
                offset=env.unwrapped.scene[self.object].cfg.frames["post_grasp"],
                asset=self.asset,
                num_envs=env.num_envs,
                device=env.device,
                max_duration=self.max_duration,
            ),
        ]

    def reset(self):
        for action in self.actions_list:
            action.reset()
