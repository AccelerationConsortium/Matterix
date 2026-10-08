# Copyright (c) 2022-2026, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""pytest collection config for source/matterix/test.

Some files in this directory are not pytest test modules at all - they are
standalone scripts that launch a real Isaac Sim instance via AppLauncher (or
drive one in a subprocess) and are meant to be run directly, e.g.::

    python source/matterix/test/test_env_step_none_action.py --headless

isaaclab/isaacsim are not installed in the hosted CPU-only unit-test job, so
merely importing these modules during pytest collection would crash the
whole job before any real test runs. Excluding them here keeps them out of
CPU collection while leaving them runnable as-is in a dedicated Isaac/GPU
job (or by hand).
"""

collect_ignore = [
    "test_env_step_none_action.py",
    "test_run_workflow_semantic_only.py",
]
