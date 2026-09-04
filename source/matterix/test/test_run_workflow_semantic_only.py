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
TurnOnHeaterCfg steps. scripts/run_workflow.py's main loop runs forever
(``while simulation_app.is_running()``), re-running episodes indefinitely in
headless mode, so this test runs it for a bounded window and checks the
captured output rather than waiting for it to exit on its own.

Usage::

    python source/matterix/test/test_run_workflow_semantic_only.py
"""

import os
import subprocess
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RUN_WORKFLOW = os.path.join(REPO_ROOT, "scripts", "run_workflow.py")

TASK = "Matterix-Test-Semantics-Heat-Transfer-Franka-v1"
WORKFLOW = "heater_only"
TIMEOUT_S = 90
MIN_EPISODES_OBSERVED = 2  # proves the inner step loop completed more than once without crashing


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
    ]
    print(f"[test] launching: {' '.join(cmd)}")

    proc = subprocess.Popen(
        cmd,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    output_lines: list[str] = []
    episode_count = 0
    deadline = time.time() + TIMEOUT_S
    try:
        while time.time() < deadline:
            line = proc.stdout.readline()
            if not line:
                if proc.poll() is not None:
                    break
                continue
            print(line, end="")
            output_lines.append(line)
            if line.startswith("EPISODE"):
                episode_count += 1
                if episode_count >= MIN_EPISODES_OBSERVED:
                    break
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=15)

    output = "".join(output_lines)

    results = {
        "no_traceback": "Traceback (most recent call last)" not in output,
        "no_nonetype_to_crash": "'NoneType' object has no attribute 'to'" not in output,
        "reached_min_episodes": episode_count >= MIN_EPISODES_OBSERVED,
    }

    print("\n=== RESULTS ===")
    for k, v in results.items():
        print(f"{k}: {'PASS' if v else 'FAIL'}")
    print(f"episodes_observed: {episode_count}")

    if not all(results.values()):
        print("\nFAILURES - see captured output above")
        return 1
    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
