# Copyright (c) 2022-2026, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression coverage for the removed rtx_raytracing_fractionalCutoutOpacity
carb setting on the base env config.

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
config never reintroduces the dead key, and translucency stays configured and
actually readable back from carb settings after a real environment
construction.

Usage::

    python source/matterix/test/test_carb_settings_translucency.py --headless
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

import matterix_tasks  # noqa: F401  registers task cfgs
from matterix_tasks.test_dev_tasks.test_franka_beaker_lift import FrankaBeakerLiftEnvTestCfg


def main() -> int:
    results: dict[str, bool] = {}

    def check(name: str, cond: bool) -> None:
        results[name] = bool(cond)

    # sim/render config is inherited unmodified from MatterixBaseEnvCfg - any concrete
    # task cfg carries the same carb_settings this test is guarding.
    cfg = FrankaBeakerLiftEnvTestCfg()
    carb_settings_cfg = cfg.sim.render.carb_settings or {}

    check("config_has_no_dead_key", "rtx_raytracing_fractionalCutoutOpacity" not in carb_settings_cfg)
    check("config_keeps_translucency", carb_settings_cfg.get("rtx_translucency_enabled") is True)

    # Live check: construct a real SimulationContext from Matterix's actual default
    # sim config and confirm the setting it owns actually reads back correctly.
    from isaaclab.sim import SimulationContext

    sim = SimulationContext(cfg.sim)
    settings = carb.settings.get_settings()

    check("translucency_reads_back_true", settings.get("/rtx/translucency/enabled") is True)

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
