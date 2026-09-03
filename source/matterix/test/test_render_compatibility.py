# Copyright (c) 2022-2026, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for Isaac Lab rendering API compatibility."""

from __future__ import annotations

import ast
import numpy as np
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


def _load_env_method(method_name, namespace=None):
    """Load a real ``MatterixBaseEnv`` method without importing Isaac Sim."""
    source_path = Path(__file__).parents[1] / "matterix" / "envs" / "matterix_base_env.py"
    module = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    env_class = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "MatterixBaseEnv")
    method = next(
        node
        for node in env_class.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method_name
    )
    method.decorator_list = []
    isolated_module = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(isolated_module)
    exec_namespace = {} if namespace is None else dict(namespace)
    exec(compile(isolated_module, str(source_path), "exec"), exec_namespace)
    return exec_namespace[method_name]


def _load_cfg_post_init():
    """Load ``MatterixBaseEnvCfg.__post_init__`` without importing Isaac Sim."""
    source_path = Path(__file__).parents[1] / "matterix" / "envs" / "matterix_base_env_cfg.py"
    module = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    cfg_class = next(
        node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "MatterixBaseEnvCfg"
    )
    post_init = next(
        node
        for node in cfg_class.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "__post_init__"
    )
    isolated_module = ast.Module(body=[post_init], type_ignores=[])
    ast.fix_missing_locations(isolated_module)
    namespace = {"PhysxCfg": SimpleNamespace}
    exec(compile(isolated_module, str(source_path), "exec"), namespace)
    return namespace["__post_init__"]


def _load_particle_runtime_policy():
    """Load the particle device policy from ``MatterixBaseEnv.__init__``."""
    source_path = Path(__file__).parents[1] / "matterix" / "envs" / "matterix_base_env.py"
    module = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    env_class = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "MatterixBaseEnv")
    init = next(node for node in env_class.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    particle_guard = next(
        node
        for node in init.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Attribute)
        and isinstance(node.test.value, ast.Name)
        and node.test.value.id == "cfg"
        and node.test.attr == "enable_particles"
    )
    policy = ast.FunctionDef(
        name="apply_particle_runtime_policy",
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg="cfg")],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=[particle_guard],
        decorator_list=[],
    )
    isolated_module = ast.Module(body=[policy], type_ignores=[])
    ast.fix_missing_locations(isolated_module)
    namespace = {}
    exec(compile(isolated_module, str(source_path), "exec"), namespace)
    return namespace["apply_particle_runtime_policy"]


def _load_workflow_main():
    """Load the ``run_workflow.main`` AST without importing Isaac Sim."""
    source_path = Path(__file__).parents[3] / "scripts" / "run_workflow.py"
    module = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    main = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    return source_path, main


def _is_record_video_guard(node):
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "args_cli"
        and node.attr == "record_video"
    )


def _disables_dataset_recorder(node):
    if not isinstance(node, ast.If) or not _is_record_video_guard(node.test):
        return False
    return any(
        isinstance(statement, ast.Assign)
        and any(
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "env_cfg"
            and target.attr == "recorders"
            for target in statement.targets
        )
        and isinstance(statement.value, ast.Constant)
        and statement.value.value is None
        for statement in node.body
    )


class IsaacLab3SimulationContext:
    """Minimal Isaac Lab 3 rendering surface; legacy render attributes are absent."""

    def __init__(self) -> None:
        self.can_render_calls = 0

    def can_render_rgb_array(self) -> bool:
        self.can_render_calls += 1
        return True

    def render(self) -> None:
        pass


class RenderCompatibilityTest(unittest.TestCase):
    def test_rgb_array_uses_isaac_lab_3_render_capability(self) -> None:
        render = _load_env_method("render", {"np": np})
        simulation_context = IsaacLab3SimulationContext()
        rgba_frame = np.full((2, 3, 4), 255, dtype=np.uint8)
        env = SimpleNamespace(
            cfg=SimpleNamespace(viewer=SimpleNamespace(resolution=(3, 2))),
            has_rtx_sensors=True,
            metadata={"render_modes": [None, "human", "rgb_array"]},
            render_mode="rgb_array",
            sim=simulation_context,
            _rgb_annotator=SimpleNamespace(get_data=lambda: rgba_frame),
        )

        frame = render(env)

        self.assertEqual(simulation_context.can_render_calls, 1)
        self.assertEqual(frame.shape, (2, 3, 3))
        np.testing.assert_array_equal(frame, rgba_frame[:, :, :3])

    def test_video_workflow_disables_unrelated_dataset_recorder(self) -> None:
        source_path, main = _load_workflow_main()
        gym_make_index = next(
            index
            for index, statement in enumerate(main.body)
            if any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "gym"
                and node.func.attr == "make"
                for node in ast.walk(statement)
            )
        )
        disable_index = next(
            (index for index, statement in enumerate(main.body) if _disables_dataset_recorder(statement)), None
        )

        self.assertIsNotNone(
            disable_index,
            f"{source_path} must disable env_cfg.recorders for --record_video so MP4-only runs do not create HDF5"
            " files",
        )
        self.assertLess(disable_index, gym_make_index)

    def test_disabled_dataset_recorder_skips_dataset_setup(self) -> None:
        setup_recorder = _load_env_method(
            "setup_recorder",
            {
                "DatasetExportMode": SimpleNamespace(EXPORT_SUCCEEDED_ONLY="succeeded"),
                "os": os,
            },
        )
        with tempfile.TemporaryDirectory() as output_dir:
            env = SimpleNamespace(
                cfg=SimpleNamespace(
                    record_path=os.path.join(output_dir, "dataset.hdf5"),
                    recorders=None,
                )
            )

            setup_recorder(env)

    def test_particle_config_uses_isaac_sim_6_physx_interface(self) -> None:
        post_init = _load_cfg_post_init()
        overwrite_gpu_setting = Mock()
        physx_module = types.ModuleType("omni.physx")
        physx_module.get_physx_interface = lambda: SimpleNamespace(overwrite_gpu_setting=overwrite_gpu_setting)
        omni_module = types.ModuleType("omni")
        omni_module.physx = physx_module
        cfg = SimpleNamespace(
            particle_systems={"fluid": object()},
            reserved_particle_systems=None,
            sim=SimpleNamespace(physics=None, use_fabric=True),
            dt=None,
            decimation=1,
            bounce_threshold_velocity=0.2,
            gpu_found_lost_aggregate_pairs_capacity=1,
            gpu_total_aggregate_pairs_capacity=2,
            friction_correlation_distance=0.025,
        )

        with patch.dict(sys.modules, {"omni": omni_module, "omni.physx": physx_module}):
            post_init(cfg)

        overwrite_gpu_setting.assert_called_once_with(1)

    def test_particle_runtime_uses_working_cpu_tensor_pipeline(self) -> None:
        apply_particle_runtime_policy = _load_particle_runtime_policy()
        cfg = SimpleNamespace(
            enable_particles=True,
            sim=SimpleNamespace(device="cuda:0", use_fabric=True),
        )

        apply_particle_runtime_policy(cfg)

        self.assertEqual(cfg.sim.device, "cpu")


if __name__ == "__main__":
    unittest.main()
