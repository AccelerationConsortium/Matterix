# Copyright (c) 2022-2025, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to run an environment with a pick and lift state machine.

The state machine is implemented in the kernel function `infer_state_machine`.
It uses the `warp` library to run the state machine in parallel on the GPU.

.. code-block:: bash

    ./isaaclab.sh -p scripts/environments/state_machine/lift_cube_sm.py --num_envs 32

"""

"""Launch Omniverse Toolkit first."""

import argparse
import matplotlib.pyplot as plt

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Pick and lift state machine for lift environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
import sys, pprint
print("Kit Python:", sys.executable)
pprint.pp(sys.path[:10])      
"""Rest everything else."""

import gymnasium as gym
import torch

from PIL import Image
from vla import load_vla
import torch

import matterix_tasks  # noqa: F401

from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
import bitsandbytes as bnb
from transformers import BitsAndBytesConfig

def main():
    # parse configuration
    env_cfg: LiftEnvCfg = parse_env_cfg(
        "Isaac-Stack-Cube-Franka-v1",
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.num_envs = args_cli.num_envs
    # create environment
    env = gym.make("Isaac-Stack-Cube-Franka-v1", cfg=env_cfg).unwrapped
    # reset environment at start
    _ = env.reset()
    q4 = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    device_map = {"model": "cuda:0",   # Llama-2
                "vlm": "cpu"}        # vision encoder + DiT on RAM


    model = load_vla(
        "CogACT/CogACT-Small",
        load_for_training=False,
        action_model_type="DiT-S",
        future_action_window_size=15,
    )
    model.vlm = model.vlm.to(torch.bfloat16)
    model.to('cuda:0').eval()

    prompt = "pick up cube"
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            table_cam_rgb = env.scene.sensors["table_cam"].data.output["rgb"].squeeze(0).cpu().numpy()
            wrist_cam_rgb = env.scene.sensors["wrist_cam"].data.output["rgb"].squeeze(0)
            print(table_cam_rgb.shape)
            # plt.imshow(table_cam_rgb.cpu())
            # plt.show()
            actions, _ = model.predict_action(
                Image.fromarray(table_cam_rgb),
                prompt,
                unnorm_key='fractal20220817_data',
                cfg_scale = 1.5,
                use_ddim = True,
                num_ddim_steps = 10,
            )

            # apply actions
            env.step(actions)

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
