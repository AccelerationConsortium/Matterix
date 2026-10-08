# Copyright (c) 2022-2026, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""End-to-end regression test: scripts/run_workflow.py must be able to execute a
pure-semantic (no agent_assets anywhere) workflow without crashing on
``action.to(env.device)`` when ``action`` is None.

This launches scripts/run_workflow.py as a real subprocess (it does its own
AppLauncher/Isaac Sim startup) against the "heater_only" workflow on
Matterix-Test-Semantics-Heat-Transfer-Franka-v1, which is composed entirely of
TurnOnHeaterCfg steps.

scripts/run_workflow.py's main loop runs forever by default (``while
simulation_app.is_running()``), so this test uses its ``--max_episodes`` flag
to make the runner stop and exit *on its own* after 2 episodes - the minimum
needed to prove the reset-and-repeat cycle works, not just that it ran once.
subprocess.communicate(timeout=...) enforces the overall timeout regardless
of whether output is flowing (unlike a manual proc.stdout.readline() loop,
which blocks indefinitely if the process goes quiet without producing more
lines), and the test asserts the process exited normally with code 0 -
proving the run actually completed successfully, rather than "we saw some
promising output before we force-killed it."

Usage::

    python source/matterix/test/test_run_workflow_semantic_only.py
"""

import os
import subprocess
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RUN_WORKFLOW = os.path.join(REPO_ROOT, "scripts", "run_workflow.py")

TASK = "Matterix-Test-Semantics-Heat-Transfer-Franka-v1"
WORKFLOW = "heater_only"
TIMEOUT_S = 90
MAX_EPISODES = 2  # minimum needed to prove the reset-and-repeat cycle works, not just a single run


def main() -> int:
    cmd = [
        sys.executable,
        RUN_WORKFLOW,
        "--task",
        TASK,
        "--workflow",
        WORKFLOW,
        "--headless",
        "--num_envs",
        "2",
        "--max_episodes",
        str(MAX_EPISODES),
    ]
    print(f"[test] launching: {' '.join(cmd)}")

    proc = subprocess.Popen(
        cmd,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    timed_out = False
    try:
        output, _ = proc.communicate(timeout=TIMEOUT_S)
    except subprocess.TimeoutExpired:
        timed_out = True
        proc.kill()
        output, _ = proc.communicate(timeout=15)

    print(output)

    episode_count = output.count("\nEPISODE ")

    results = {
        "did_not_time_out": not timed_out,
        "exited_normally": proc.returncode == 0,
        "no_traceback": "Traceback (most recent call last)" not in output,
        "no_nonetype_to_crash": "'NoneType' object has no attribute 'to'" not in output,
        "ran_expected_episode_count": episode_count == MAX_EPISODES,
    }

    print("\n=== RESULTS ===")
    for k, v in results.items():
        print(f"{k}: {'PASS' if v else 'FAIL'}")
    print(f"episodes_observed: {episode_count}, returncode: {proc.returncode}")

    if not all(results.values()):
        print("\nFAILURES - see captured output above")
        return 1
    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
