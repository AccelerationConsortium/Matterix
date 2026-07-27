# Copyright (c) 2022-2026, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run an environment with zero action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Zero agent for matterix environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=None, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--record_video", action="store_true", default=False, help="Record a video of each episode.")
parser.add_argument(
    "--video_dir",
    type=str,
    default="out/videos",
    help="Directory to save recorded videos (default: out/videos).",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

if args_cli.record_video and args_cli.headless and not args_cli.enable_cameras:
    parser.error("--record_video in headless mode requires --enable_cameras (the RTX renderer must be loaded for frame capture).")

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import datetime
import os

import gymnasium as gym
import torch

import matterix_tasks  # noqa: F401

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def main():
    """Zero actions agent with matterix environment."""
    # parse configuration
    use_fabric = None if args_cli.disable_fabric is None else not args_cli.disable_fabric
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=use_fabric)
    # create environment
    print("=============================================")
    print("[INFO]: Environment created successfully.")
    print("env_cfg num_envs:", env_cfg.scene.num_envs)
    print("args_cli num_envs:", args_cli.num_envs)

    render_mode = "rgb_array" if args_cli.record_video else None
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)
    print("env_cfg num_envs:", env_cfg.scene.num_envs)
    print("args_cli num_envs:", args_cli.num_envs)
    print("=============================================")

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    run_ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    task_slug = (args_cli.task or "task").replace("/", "_")
    episode_count = 0

    # reset environment
    env.reset()
    episode_count += 1
    if args_cli.record_video:
        env.unwrapped.start_recording()

    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # compute zero actions
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            # apply actions
            observations, reward, terminated, truncated, info = env.step(actions)
            print("[zero agent] observations:", observations)

            if args_cli.record_video and bool((terminated | truncated).any()):
                video_path = os.path.join(
                    args_cli.video_dir, f"{task_slug}_ep{episode_count}_{run_ts}.mp4"
                )
                env.unwrapped.save_video(video_path)
                print(f"[INFO]: Video saved to {video_path}")
                episode_count += 1
                env.unwrapped.start_recording()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
