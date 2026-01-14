#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
List available workflows for MATteRIX environments.

This script lists workflow definitions from environment configs.
Note: Requires Isaac Sim to be initialized to load environment configs.

Usage:
    python scripts/list_workflows.py --task Matterix-Test-Beaker-Lift-Franka-v1
    python scripts/list_workflows.py --all
"""

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app


"""Rest everything follows."""

import argparse
import gymnasium as gym
import sys

import matterix_tasks  # noqa: F401

# Add argparse arguments
parser = argparse.ArgumentParser(
    description="List available workflows for MATteRIX environments (no simulation required)."
)
parser.add_argument(
    "--task",
    type=str,
    default=None,
    help="Environment/task name to list workflows for (e.g., Matterix-Test-Beaker-Lift-Franka-v1)",
)
parser.add_argument(
    "--all",
    action="store_true",
    help="List workflows for all registered environments",
)

args = parser.parse_args()


def get_env_config_class(task_name: str):
    """Get the environment config class from Gymnasium registry.

    Args:
        task_name: Name of the environment/task

    Returns:
        Environment config class (not instance)
    """
    try:
        # Get the environment spec from Gymnasium registry
        env_spec = gym.spec(task_name)

        # The kwargs should contain env_cfg_entry_point which points to the config class
        if "env_cfg_entry_point" in env_spec.kwargs:
            env_cfg_entry_point = env_spec.kwargs["env_cfg_entry_point"]

            # If it's a string, we need to import it
            if isinstance(env_cfg_entry_point, str):
                module_path, class_name = env_cfg_entry_point.rsplit(".", 1)
                import importlib

                module = importlib.import_module(module_path)
                return getattr(module, class_name)
            else:
                # It's already a class
                return env_cfg_entry_point

        raise ValueError(f"Could not find config for {task_name}")
    except Exception as e:
        raise ValueError(f"Error loading config for {task_name}: {e}")


def list_workflows_for_task(task_name: str) -> None:
    """List all workflows for a given task.

    Args:
        task_name: Name of the environment/task (e.g., "Matterix-Test-Beaker-Lift-Franka-v1")
    """
    try:
        # Get the config class
        env_cfg_class = get_env_config_class(task_name)

        if env_cfg_class is None:
            print(f"\n{task_name}")
            print("  Could not load environment config.")
            return

        # The @configclass decorator removes non-annotated class variables, but they become
        # instance attributes when you create an instance. So we need to create an instance
        # to access workflows (same as test_sm.py does with parse_env_cfg)
        try:
            env_cfg_instance = env_cfg_class()
            workflows = getattr(env_cfg_instance, "workflows", None)
        except Exception as e:
            print(f"Warning: Could not instantiate config: {e}")
            print("Trying to access workflows as class attribute...")
            workflows = getattr(env_cfg_class, "workflows", None)

        if not workflows or not hasattr(workflows, "items"):
            print(f"\n{task_name}")
            print("  No workflows defined.")
            return

        print("=" * 80)
        print(f"{task_name}")
        print("=" * 80)

        for workflow_name, workflow_value in workflows.items():
            # Handle two formats:
            # 1. Dictionary format: {"description": "...", "actions": [...]}
            # 2. Direct action config: PickObjectCfg(...)
            if isinstance(workflow_value, dict):
                description = workflow_value.get("description", "No description")
                actions = workflow_value.get("actions", [])
            else:
                description = getattr(workflow_value, "description", "No description")
                actions = [workflow_value]

            action_types = ", ".join([type(action).__name__ for action in actions])
            print(f"\n        name: {workflow_name}")
            print(f" description: {description}")
            print(f"     actions: {action_types}")

        print("=" * 80)

    except Exception as e:
        print(f"\n{task_name}")
        print(f"  Error: {e}")


def list_all_workflows() -> None:
    """List workflows for all registered MATteRIX environments."""
    # Get all registered Gymnasium environments
    all_envs = gym.envs.registry.keys()

    # Filter for MATteRIX environments
    matterix_envs = [env for env in all_envs if env.startswith("Matterix-")]

    if not matterix_envs:
        print("No MATteRIX environments found in Gymnasium registry.")
        print("Make sure matterix_tasks is properly installed.")
        sys.exit(1)

    print(f"\nFound {len(matterix_envs)} MATteRIX environment(s):")
    print("=" * 80)

    for env_name in sorted(matterix_envs):
        list_workflows_for_task(env_name)


def main():
    if args.all:
        list_all_workflows()
    elif args.task:
        list_workflows_for_task(args.task)
    else:
        print("Error: Please specify --task or --all")
        print("Examples:")
        print("  python scripts/list_workflows.py --task Matterix-Test-Beaker-Lift-Franka-v1")
        print("  python scripts/list_workflows.py --all")
        sys.exit(1)


if __name__ == "__main__":
    main()
    # close sim app
    simulation_app.close()
