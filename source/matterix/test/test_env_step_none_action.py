# Copyright (c) 2022-2026, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac runtime regression test for MatterixBaseEnv.step(None, semantic_actions=...).

Requires a live Isaac Sim instance (unlike the matterix_sm-only StateMachine
tests, this exercises the real ActionManager/SemanticManager against a real
scene), so it follows this repo's existing AppLauncher-first test convention
(see test_video_recording.py) rather than plain pytest.

Verifies:
  1. action_manager.process_action() is NOT called when action=None.
  2. The semantic action is applied exactly once (heater turns on in obs).
  3. action_manager.action (the buffer apply_action() reads from every
     decimation substep) is byte-for-byte unchanged across a None step -
     i.e. no zero/stale controller targets get applied.
  4. Both transitions:
     a. immediately after env.reset() -> semantic-only step
     b. a real robot action -> semantic-only step, where the robot must hold
        its previously-commanded target exactly.

Usage::

    python source/matterix/test/test_env_step_none_action.py --headless
"""

import argparse

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--task",
    type=str,
    default="Matterix-Test-Semantics-Heat-Transfer-Franka-v1",
    help="Task to run this regression test against (needs a heater/semantic asset).",
)

from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import sys

import torch

import gymnasium as gym

import matterix_tasks  # noqa: F401  registers gym envs
from isaaclab.managers.action_manager import ActionManager
from isaaclab_tasks.utils import parse_env_cfg
from matterix_sm.semantic_info import SemanticInfo


def main() -> int:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=2)
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    results: dict[str, bool] = {}

    def check(name: str, cond: bool) -> None:
        results[name] = bool(cond)

    # Spy on the class method rather than the instance, since ActionManager may not
    # support arbitrary instance attribute assignment.
    call_count = {"n": 0}
    orig_process_action = ActionManager.process_action

    def spy_process_action(self, action):
        call_count["n"] += 1
        return orig_process_action(self, action)

    ActionManager.process_action = spy_process_action

    semantic = [
        SemanticInfo(
            type="IsHeaterOn",
            asset_name="ika_plate",
            value=True,
            additional_info={"target_temperature": 373.15},
        )
    ]

    try:
        # === Scenario A: immediately after reset -> semantic-only action ===
        obs, _ = env.reset()
        call_count["n"] = 0
        buf_before = env.action_manager.action.clone()

        obs, _, _, _, _ = env.step(None, semantic_actions=semantic)

        check("A_process_action_not_called", call_count["n"] == 0)
        check("A_action_buffer_unchanged", torch.equal(buf_before, env.action_manager.action))
        check("A_heater_turned_on", bool(obs["policy"]["ika_plate_is_heater_on"].all().item()))

        # === Scenario B: real robot action -> semantic action, must hold previous target ===
        obs, _ = env.reset()
        call_count["n"] = 0

        action_dim = env.action_manager.total_action_dim
        real_action = torch.full((env.num_envs, action_dim), 0.1234, device=env.device)
        obs, _, _, _, _ = env.step(real_action)
        check("B_process_action_called_on_real_step", call_count["n"] == 1)
        target_after_real = env.action_manager.action.clone()

        call_count["n"] = 0
        obs, _, _, _, _ = env.step(None, semantic_actions=semantic)
        check("B_process_action_not_called_on_none_step", call_count["n"] == 0)
        check("B_target_held_exactly", torch.equal(target_after_real, env.action_manager.action))
        check("B_heater_turned_on", bool(obs["policy"]["ika_plate_is_heater_on"].all().item()))
    finally:
        ActionManager.process_action = orig_process_action
        env.close()

    print("\n=== RESULTS ===")
    for k, v in results.items():
        print(f"{k}: {'PASS' if v else 'FAIL'}")

    failures = [k for k, v in results.items() if not v]
    if failures:
        print(f"\nFAILURES: {failures}")
        return 1
    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    rc = main()
    simulation_app.close()
    sys.exit(rc)
