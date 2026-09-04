# Copyright (c) 2022-2026, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression coverage for the removed rtx_raytracing_fractionalCutoutOpacity
carb setting on the base env config, and its replacement on the "pathtracing"
video-recording preset.

Background: rtx_raytracing_fractionalCutoutOpacity is a raytracing-namespaced
setting. SimulationContext.__init__() (which applies the base env's
carb_settings) always runs before the renderer switches into any raytracing/
path-tracing mode, so the raytracing extension that owns this setting hasn't
registered it yet at that point -- confirmed identically on
isaaclab==2.3.0/isaacsim==5.1.0.0 and isaaclab==3.0.0b2.post1/isaacsim==6.0.1.0.
It's not a renamed-or-retired-upstream setting; it was applied at a lifecycle
stage where it can never take effect in the base (non-raytraced) render path,
on any version this project has targeted. isaaclab==2.3.0 additionally
validated carb_settings keys against the registry and raised ValueError for
one that wasn't registered yet (the original crash, #23); isaaclab==3.0.0b2.post1
dropped that validation, so the same key is a silent no-op there instead.
Neither way did it ever take effect on the base config, so it's removed there.

IsaacLab's own validation behavior is NOT something this test pins down, since
it has already changed shape once across versions for reasons outside
Matterix's control. This test instead pins down what Matterix owns: the base
config never reintroduces the dead key there, translucency stays configured,
and -- since the setting genuinely does exist once raytracing mode is
engaged -- the "pathtracing" preset (the one place it can take effect) sets it
explicitly and it reads back correctly after a real path-traced render.

Usage::

    python source/matterix/test/test_carb_settings_translucency.py --headless --enable_cameras
"""

import argparse

parser = argparse.ArgumentParser(description=__doc__)

from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import sys

import carb.settings

import gymnasium as gym
import torch

import matterix_tasks  # noqa: F401  registers task cfgs
from isaaclab_tasks.utils import parse_env_cfg
from matterix_tasks.test_dev_tasks.test_franka_beaker_lift import FrankaBeakerLiftEnvTestCfg

TASK = "Matterix-Test-Beaker-Lift-Franka-v1"


def main() -> int:
    results: dict[str, bool] = {}

    def check(name: str, cond: bool) -> None:
        results[name] = bool(cond)

    # --- Base config: the dead key must never reappear, translucency must stay set ---
    # sim/render config is inherited unmodified from MatterixBaseEnvCfg - any concrete
    # task cfg carries the same carb_settings this test is guarding.
    base_cfg = FrankaBeakerLiftEnvTestCfg()
    carb_settings_cfg = base_cfg.sim.render.carb_settings or {}

    check("base_config_has_no_dead_key", "rtx_raytracing_fractionalCutoutOpacity" not in carb_settings_cfg)
    check("base_config_keeps_translucency", carb_settings_cfg.get("rtx_translucency_enabled") is True)

    from isaaclab.sim import SimulationContext

    sim = SimulationContext(base_cfg.sim)
    settings = carb.settings.get_settings()

    check("translucency_reads_back_true", settings.get("/rtx/translucency/enabled") is True)
    # NOTE: deliberately not asserting on whether /rtx/raytracing/fractionalCutoutOpacity
    # is registered at this point - that depends on which raytracing/RTX extensions
    # happened to load (e.g. --enable_cameras alone can pull enough of the stack in
    # eagerly to register it), not on anything Matterix controls. See module docstring.
    SimulationContext.clear_instance()

    # --- Pathtracing preset: the one place the setting can actually take effect ---
    pt_cfg = parse_env_cfg(TASK, device=args_cli.device, num_envs=1)
    pt_cfg.prepare_for_video_rec(resolution=(640, 480), render="pathtracing")

    check(
        "pathtracing_preset_sets_cutout_opacity",
        pt_cfg.sim.render.carb_settings.get("/rtx/pathtracing/fractionalCutoutOpacity") is True,
    )

    env = gym.make(TASK, cfg=pt_cfg, render_mode="rgb_array").unwrapped
    env.reset()
    with torch.inference_mode():
        action = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)
        for _ in range(10):
            env.step(action)
            env.render()

    settings = carb.settings.get_settings()
    check("pathtracing_rendermode_engaged", settings.get("/rtx/rendermode") in ("PathTracing", "RaytracedLighting", "RealTimePathTracing"))
    check(
        "pathtracing_cutout_opacity_reads_back_true",
        settings.get("/rtx/pathtracing/fractionalCutoutOpacity") is True,
    )
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
