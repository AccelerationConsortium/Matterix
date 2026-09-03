# Copyright (c) 2022-2026, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for Isaac Lab 3.0 XYZW quaternion math."""

import importlib.util
import math
import torch
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "matterix_sm" / "math_utils.py"
SPEC = importlib.util.spec_from_file_location("matterix_sm_math_utils", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MATH_UTILS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MATH_UTILS)
quat_mul = MATH_UTILS.quat_mul
quat_rotate = MATH_UTILS.quat_rotate


def test_identity_quaternion_preserves_vector() -> None:
    """XYZW identity must not rotate a vector."""
    identity = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    vector = torch.tensor([[1.0, -2.0, 3.0]])

    torch.testing.assert_close(quat_rotate(identity, vector), vector)


def test_positive_z_rotation_uses_xyzw_order() -> None:
    """A positive 90-degree Z rotation must map +X to +Y."""
    half_sqrt = math.sqrt(0.5)
    rotation = torch.tensor([[0.0, 0.0, half_sqrt, half_sqrt]])
    vector = torch.tensor([[1.0, 0.0, 0.0]])

    torch.testing.assert_close(
        quat_rotate(rotation, vector),
        torch.tensor([[0.0, 1.0, 0.0]]),
        atol=1.0e-6,
        rtol=1.0e-6,
    )


def test_identity_is_neutral_for_xyzw_multiplication() -> None:
    """Hamilton multiplication must preserve an XYZW rotation."""
    half_sqrt = math.sqrt(0.5)
    rotation = torch.tensor([[half_sqrt, 0.0, 0.0, half_sqrt]])
    identity = torch.tensor([[0.0, 0.0, 0.0, 1.0]])

    torch.testing.assert_close(quat_mul(rotation, identity), rotation)
    torch.testing.assert_close(quat_mul(identity, rotation), rotation)
