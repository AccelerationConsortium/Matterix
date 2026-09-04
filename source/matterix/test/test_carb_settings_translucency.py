# Copyright (c) 2022-2026, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression coverage for the removed rtx_raytracing_fractionalCutoutOpacity
carb setting.

Note on scope: IsaacLab's own validation of unregistered carb_settings keys is
NOT something this test can pin down, because it has already changed shape
once across the versions this project has targeted -- isaaclab==2.3.0 raised
ValueError for an unregistered key (the original bug), while
isaaclab==3.0.0b2.post1 silently no-ops instead. Asserting on that upstream
behavior would make this test pass/fail depending on which IsaacLab version
it happens to run against, for reasons entirely outside Matterix's control.

Instead this test pins down what Matterix actually owns: the config itself
must never reintroduce the dead key, translucency must stay configured, and
the value that setting maps to must actually be readable back from carb
settings after a real environment construction.

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

OBSOLETE_KEYS = (
    "rtx_raytracing_fractionalCutoutOpacity",
    "/rtx/raytracing/fractionalCutoutOpacity",
    "/rtx/pathtracing/fractionalCutoutOpacity",
)


def main() -> int:
    results: dict[str, bool] = {}

    def check(name: str, cond: bool) -> None:
        results[name] = bool(cond)

    # sim/render config is inherited unmodified from MatterixBaseEnvCfg - any concrete
    # task cfg carries the same carb_settings this test is guarding.
    cfg = FrankaBeakerLiftEnvTestCfg()
    carb_settings_cfg = cfg.sim.render.carb_settings or {}

    check(
        "config_has_no_obsolete_key",
        all(key not in carb_settings_cfg for key in ("rtx_raytracing_fractionalCutoutOpacity",)),
    )
    check("config_keeps_translucency", carb_settings_cfg.get("rtx_translucency_enabled") is True)

    # Live check: construct a real SimulationContext from Matterix's actual default
    # sim config and confirm the setting it owns actually reads back correctly.
    from isaaclab.sim import SimulationContext

    sim = SimulationContext(cfg.sim)
    settings = carb.settings.get_settings()

    check("translucency_reads_back_true", settings.get("/rtx/translucency/enabled") is True)
    check(
        "no_obsolete_setting_registered",
        all(settings.get(key) is None for key in ("/rtx/raytracing/fractionalCutoutOpacity",)),
    )

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
