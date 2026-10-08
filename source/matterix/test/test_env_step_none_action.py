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
  2. The semantic action is applied exactly once - IsHeaterOn.set_value() (the
     actual application point, not just the resulting obs value) is
     call-counted directly.
  3. env.scene["robot"].data.joint_pos_target - the real PD-controller target
     PhysX applies every decimation substep - is not dropped to zero and
     contains no NaN/Inf after a None step. (action_manager.action only
     reflects the raw input buffer before per-term processing, so it can't
     prove anything about the actuator target itself.) Note: this target is
     NOT expected to be byte-for-byte frozen across steps - the Franka arm's
     IK controller continuously re-solves against the robot's live, still-
     converging joint state every step, so it drifts by a similar amount
     step-to-step whether or not a new action was given (measured: ~0.24-0.27
     rad max-abs-diff either way). What actually indicates a real problem is
     the target getting zeroed out or corrupted, which is what's checked here.
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

import gymnasium as gym
import sys
import torch

import matterix_tasks  # noqa: F401  registers gym envs
from matterix.managers.semantics.primitive_semantics.heat_transfer.is_heater_on import IsHeaterOn
from matterix_sm.semantic_info import SemanticInfo

from isaaclab.managers.action_manager import ActionManager
from isaaclab_tasks.utils import parse_env_cfg


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

    # Spy on the actual semantic application point (not just the resulting obs
    # value), so we can assert it fires exactly once per semantic step.
    heater_call_count = {"n": 0}
    orig_set_value = IsHeaterOn.set_value

    def spy_set_value(self, semantic_info):
        heater_call_count["n"] += 1
        return orig_set_value(self, semantic_info)

    IsHeaterOn.set_value = spy_set_value

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
        heater_call_count["n"] = 0
        buf_before = env.action_manager.action.clone()

        obs, _, _, _, _ = env.step(None, semantic_actions=semantic)
        joint_target = env.scene["robot"].data.joint_pos_target

        check("A_process_action_not_called", call_count["n"] == 0)
        check("A_action_buffer_unchanged", torch.equal(buf_before, env.action_manager.action))
        check("A_joint_target_not_zero", not torch.allclose(joint_target, torch.zeros_like(joint_target)))
        check("A_joint_target_finite", bool(torch.isfinite(joint_target).all()))
        check("A_heater_set_value_called_once", heater_call_count["n"] == 1)
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
        heater_call_count["n"] = 0
        obs, _, _, _, _ = env.step(None, semantic_actions=semantic)
        joint_target = env.scene["robot"].data.joint_pos_target

        check("B_process_action_not_called_on_none_step", call_count["n"] == 0)
        check("B_target_held_exactly", torch.equal(target_after_real, env.action_manager.action))
        check("B_joint_target_not_zero", not torch.allclose(joint_target, torch.zeros_like(joint_target)))
        check("B_joint_target_finite", bool(torch.isfinite(joint_target).all()))
        check("B_heater_set_value_called_once", heater_call_count["n"] == 1)
        check("B_heater_turned_on", bool(obs["policy"]["ika_plate_is_heater_on"].all().item()))
    finally:
        ActionManager.process_action = orig_process_action
        IsHeaterOn.set_value = orig_set_value
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
