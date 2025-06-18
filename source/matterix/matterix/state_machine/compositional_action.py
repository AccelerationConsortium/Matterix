from .actions import *
import torch

class CompositionalAction():

    def __init__(self, asset = "robot", num_envs : int = 1, device: torch.device =  torch.device("cuda" if torch.cuda.is_available() else "cpu"), max_duration: int = 100):
        self.actions_list = []
        self.asset = asset
        self.num_envs = num_envs
        self.device = device,
        self.max_duration = max_duration
    def initialize(self, env):
         raise NotImplementedError

class PickObject(CompositionalAction):
    
    def __init__(self, object, asset = "robot", num_envs : int = 1, device: torch.device =  torch.device("cuda" if torch.cuda.is_available() else "cpu"), max_duration: int = 100):
        super().__init__(asset, num_envs, device, max_duration)
        self.object = object

    def initialize(self, env):
        self.actions_list = [
            MoveToFrame(object=self.object, frame="pre_grasp", asset=self.asset, num_envs=env.num_envs, device=env.device, max_duration=self.max_duration),
            OpenGripper(asset=self.asset, num_envs=env.num_envs, device=env.device),
            MoveToFrame(object=self.object, frame="grasp", asset=self.asset, num_envs=env.num_envs, device=env.device, max_duration=self.max_duration),
            CloseGripper(asset=self.asset, num_envs=env.num_envs, device=env.device),
            MoveRelative(offset=env.unwrapped.scene[self.object].cfg.frames["post_grasp"], asset=self.asset, num_envs=env.num_envs, device=env.device, max_duration=self.max_duration)
            ]
    def reset(self):
         for action in self.actions_list:
              action.reset()