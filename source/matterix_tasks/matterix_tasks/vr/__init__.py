import gymnasium as gym
import os

from isaaclab_tasks.manager_based.manipulation.stack.config.franka import agents
from . import stack_ik_rel_env_cfg_skillgen

gym.register(
    id="Matterix-Test-VR-Franka-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stack_ik_rel_env_cfg_skillgen:FrankaCubeStackSkillgenEnvCfg",
        # "robomimic_bc_cfg_entry_point": f"{agents.__name__}:robomimic/bc_rnn_low_dim.json",
    },
    disable_env_checker=True,
)