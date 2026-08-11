# Copyright (c) 2022-2026, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Run a workflow using the MATteRIX state machine system.

The state machine orchestrates sequential actions across parallel environments on the GPU.

Usage:
    ./matterix.sh -p scripts/run_workflow.py --task Matterix-Test-Beaker-Lift-Franka-v1 --workflow pickup_beaker
    ./matterix.sh -p scripts/run_workflow.py --task Matterix-Test-Beaker-Lift-Franka-v1 --workflow pickup_beaker --record_video
"""

"""Launch Omniverse Toolkit first."""

import argparse

from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Run state machine workflows for MATteRIX environments.")
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=1,
    help="Number of parallel environments to simulate.",
)
parser.add_argument(
    "--task",
    type=str,
    default="Matterix-Test-Beaker-Lift-Franka-v1",
    help="Environment/task name.",
)
parser.add_argument("--workflow", type=str, default="pickup_beaker", help="Name of the workflow to run.")
parser.add_argument("--record_video", action="store_true", default=False, help="Record a video of each episode.")
parser.add_argument(
    "--video_dir",
    type=str,
    default="out/videos",
    help="Directory to save recorded videos (default: out/videos).",
)
parser.add_argument("--record_path", type=str, default=None, help="Optional unique HDF5 recorder path for this run.")
parser.add_argument("--episodes", type=int, default=0, help="Stop after this many episodes; 0 keeps the existing continuous behavior.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.record_video and args_cli.headless and not args_cli.enable_cameras:
    parser.error("--record_video in headless mode requires --enable_cameras (the RTX renderer must be loaded for frame capture).")

# Launch omniverse app
# Forward the complete parsed launcher configuration so --livestream reaches WebRTC.
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything else."""

import datetime
import os

import gymnasium as gym
import torch

import matterix_tasks  # noqa: F401
from matterix_sm import StateMachine

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


def main():
    # Parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    if args_cli.record_path is not None:
        env_cfg.record_path = args_cli.record_path

    # Validate workflow exists
    if not hasattr(env_cfg, "workflows") or not env_cfg.workflows:
        raise ValueError(f"No workflows defined for {args_cli.task}!")

    if args_cli.workflow not in env_cfg.workflows:
        available = list(env_cfg.workflows.keys())
        raise ValueError(
            f"Workflow '{args_cli.workflow}' not found. Available workflows: {available}. "
            f"Use 'python scripts/list_workflows.py --task {args_cli.task}' to see details."
        )

    # Extract workflow actions
    workflow_value = env_cfg.workflows[args_cli.workflow]
    if isinstance(workflow_value, dict):
        description = workflow_value.get("description", "No description")
        actions = workflow_value.get("actions", [])
    elif isinstance(workflow_value, list):
        description = "No description"
        actions = workflow_value
    else:
        description = getattr(workflow_value, "description", "No description")
        actions = [workflow_value]

    print(f"\nTask: {args_cli.task}")
    print(f"Workflow: '{args_cli.workflow}'")
    print(f"Description: {description}\n")

    # Create environment and state machine
    render_mode = "rgb_array" if args_cli.record_video else None
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode).unwrapped
    env.reset()

    # Create state machine with required parameters from environment
    sm = StateMachine(num_envs=env.num_envs, dt=env.step_dt, device=env.device)
    sm.set_action_sequence(actions)

    # Timestamp shared across all episodes of this run
    run_ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    task_slug = args_cli.task.replace("/", "_")
    workflow_slug = args_cli.workflow.replace("/", "_")

    episode_count = 0

    # Main simulation loop
    while simulation_app.is_running():
        with torch.inference_mode():
            obs, _ = env.reset()
            sm.reset()
            episode_count += 1
            step_count = 0

            print(f"\n{'=' * 80}")
            print(f"EPISODE {episode_count}")
            print(f"{'=' * 80}\n")

            if args_cli.record_video:
                env.start_recording()

            # Run until workflow completes or fails
            while not (sm.action_sequence_success | sm.action_sequence_failure).all():
                action, semantic_actions = sm.step(obs)
                action = action.to(env.device)
                obs, _, terminated, truncated, _ = env.step(action, semantic_actions=semantic_actions)
                step_count += 1

                # Reset SM for any envs the environment auto-reset (episode timeout/termination)
                reset_ids = (terminated | truncated).nonzero(as_tuple=False).flatten()
                if reset_ids.numel() > 0:
                    sm.reset_envs(reset_ids)

                # Print status every 50 steps
                if step_count % 50 == 0:
                    sm.print_status(step=step_count, episode=episode_count)

            sm.print_status(step=step_count, episode=episode_count)
            if args_cli.episodes > 0 and episode_count >= args_cli.episodes:
                break

            if args_cli.record_video:
                video_path = os.path.join(
                    args_cli.video_dir, f"{task_slug}_{workflow_slug}_ep{episode_count}_{run_ts}.mp4"
                )
                env.save_video(video_path)
                print(f"[INFO]: Video saved to {video_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
