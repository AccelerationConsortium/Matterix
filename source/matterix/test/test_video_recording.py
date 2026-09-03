# Copyright (c) 2022-2026, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test ``MatterixBaseEnv`` video recording across render presets.

Each render preset requires a fresh sim app (carb settings baked at init),
so multi-mode runs dispatch each mode as a subprocess.

Uses the beaker-lift task with the ``pickup_beaker`` workflow so the
state machine drives actual robot motion.

Usage::

    # Fast smoke test (one mode, 720p):
    python source/matterix/test/test_video_recording.py --headless --enable_cameras

    # All four render presets at 4K for visual comparison:
    python source/matterix/test/test_video_recording.py --headless --enable_cameras --all_modes --resolution 3840 2160

    # Single preset at custom resolution:
    python source/matterix/test/test_video_recording.py --headless --enable_cameras --mode pathtracing --resolution 3840 2160
"""

import argparse
import os
import subprocess
import sys
import time

MODES = ("default", "balanced", "quality", "pathtracing")

DEFAULT_TASK = "Matterix-Test-Beaker-Lift-Franka-v1"
DEFAULT_WORKFLOW = "pickup_beaker"
# Closer camera on the default (7.5,7.5,7.5)->(0,0,0) diagonal, centered on workspace
DEFAULT_EYE = (1.8, 1.8, 1.8)
DEFAULT_LOOKAT = (0.3, 0.0, 0.3)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        type=str,
        default=DEFAULT_TASK,
        help="Matterix task ID to render.",
    )
    parser.add_argument(
        "--workflow",
        type=str,
        default=DEFAULT_WORKFLOW,
        help="Workflow name to run (from env_cfg.workflows).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=MODES,
        help="Single render preset to run.",
    )
    parser.add_argument(
        "--all_modes",
        action="store_true",
        help="Run all four render presets (subprocess each).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Where mp4 files are written. Defaults to <MATTERIX_PATH>/out/render_modes.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=300,
        help="Max env steps (workflow may finish earlier).",
    )
    parser.add_argument(
        "--resolution",
        nargs=2,
        type=int,
        default=[1280, 720],
        metavar=("W", "H"),
        help="Viewport resolution.",
    )

    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)

    return parser.parse_args()


def _run_all_modes(args) -> int:
    """Re-launch this script once per mode and return the worst exit code."""
    passthrough = []
    skip = False
    for a in sys.argv[1:]:
        if skip:
            skip = False
            continue
        if a == "--mode":
            skip = True
            continue
        if a.startswith("--mode="):
            continue
        if a == "--all_modes":
            continue
        passthrough.append(a)

    worst = 0
    for mode in MODES:
        cmd = [sys.executable, sys.argv[0], "--mode", mode, *passthrough]
        print(f"\n[runner] === {mode} ===")
        print(f"[runner] {' '.join(cmd)}")
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            print(f"[runner] {mode} failed with rc={rc}", file=sys.stderr)
            worst = rc
    return worst


def run_single_mode(
    task: str, workflow: str, mode: str, resolution: tuple[int, int], max_steps: int, out_dir: str, args
) -> str:
    """Launch sim, run workflow with video recording, save mp4."""
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import torch

    import matterix_tasks  # noqa: F401
    from matterix_sm import StateMachine

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg(task, device=args.device, num_envs=1)
    env_cfg.prepare_for_video_rec(
        resolution=resolution,
        render=mode,
        eye=DEFAULT_EYE,
        lookat=DEFAULT_LOOKAT,
    )

    env = gym.make(task, cfg=env_cfg, render_mode="rgb_array").unwrapped

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{mode}.mp4")
    print(
        f"[capture] task={task} workflow={workflow} mode={mode} "
        f"resolution={resolution} max_steps={max_steps} -> {out_path}",
        flush=True,
    )

    workflow_cfg = env_cfg.workflows[workflow]
    actions_seq = workflow_cfg if isinstance(workflow_cfg, list) else getattr(workflow_cfg, "actions", [workflow_cfg])

    sm = StateMachine(num_envs=env.num_envs, dt=env.step_dt, device=env.device)
    sm.set_action_sequence(actions_seq)

    env.start_recording()
    obs, _ = env.reset()
    sm.reset()

    t_start = time.perf_counter()
    step_count = 0
    for i in range(max_steps):
        with torch.inference_mode():
            action, semantic_actions = sm.step(obs)
            action = action.to(env.device)
            obs, _, terminated, truncated, _ = env.step(action, semantic_actions=semantic_actions)
        step_count = i + 1

        if (sm.action_sequence_success | sm.action_sequence_failure).all():
            print(f"[capture] workflow finished at step {step_count}", flush=True)
            break
        if step_count % 50 == 0:
            print(f"[capture] step {step_count}/{max_steps}", flush=True)

    t_sim = time.perf_counter() - t_start

    t_save = time.perf_counter()
    env.save_video(out_path)
    t_save = time.perf_counter() - t_save

    file_size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"[capture] wrote {out_path} ({file_size_mb:.1f} MB)", flush=True)
    print(
        f"[timing] mode={mode} resolution={resolution} steps={step_count} "
        f"sim={t_sim:.1f}s encode={t_save:.1f}s total={t_sim + t_save:.1f}s",
        flush=True,
    )

    simulation_app  # silence linter
    os._exit(0)


if __name__ == "__main__":
    args = _parse_args()

    if args.out_dir is None:
        matterix_path = os.environ.get("MATTERIX_PATH", os.getcwd())
        args.out_dir = os.path.join(matterix_path, "out", "render_modes")

    if args.all_modes:
        sys.exit(_run_all_modes(args))

    mode = args.mode or "balanced"
    run_single_mode(
        task=args.task,
        workflow=args.workflow,
        mode=mode,
        resolution=tuple(args.resolution),
        max_steps=args.max_steps,
        out_dir=args.out_dir,
        args=args,
    )
