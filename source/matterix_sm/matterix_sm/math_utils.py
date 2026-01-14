# Copyright (c) 2022-2026, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mathematical utility functions for quaternion operations and transformations.

This module provides pure mathematical utilities used by primitive actions,
particularly for quaternion arithmetic and coordinate frame transformations.
"""

import torch


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector v by quaternion q.

    Args:
        q: (N, 4) quaternions in (w, x, y, z) format
        v: (N, 3) vectors to rotate

    Returns:
        (N, 3) rotated vectors
    """
    # Extract quaternion components
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    vx, vy, vz = v[:, 0], v[:, 1], v[:, 2]

    # Compute rotation using quaternion formula
    # v' = v + 2*w*(q_vec x v) + 2*(q_vec x (q_vec x v))
    # where q_vec = (x,y,z)
    uvx = y * vz - z * vy
    uvy = z * vx - x * vz
    uvz = x * vy - y * vx

    uuvx = y * uvz - z * uvy
    uuvy = z * uvx - x * uvz
    uuvz = x * uvy - y * uvx

    result_x = vx + 2 * (w * uvx + uuvx)
    result_y = vy + 2 * (w * uvy + uuvy)
    result_z = vz + 2 * (w * uvz + uuvz)

    return torch.stack([result_x, result_y, result_z], dim=-1)


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Multiply two quaternions (Hamilton product).

    Args:
        q1: (N, 4) quaternions in (w, x, y, z) format
        q2: (N, 4) quaternions in (w, x, y, z) format

    Returns:
        (N, 4) product quaternions in (w, x, y, z) format
    """
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack([w, x, y, z], dim=-1)
