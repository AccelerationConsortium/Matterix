# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import torch

from isaaclab.devices.device_base import DeviceBase
from isaaclab.devices.retargeter_base import RetargeterBase, RetargeterCfg


class GripperVRRetargeter(RetargeterBase):
    """Retargeter specifically for gripper control based on hand tracking data.

    This retargeter grabs the motion controller (XR Remote)'s primary key value to determine
    whether the gripper should be open or closed. It includes hysteresis to prevent rapid
    toggling between states when the pincher distance is near the thresholds.

    Features:
    - Tracks controller input
    - Implements hysteresis for stable gripper control
    - Outputs boolean command (True = close gripper, False = open gripper)
    """

    GRIPPER_CLOSE_METERS: Final[float] = 0.03
    GRIPPER_OPEN_METERS: Final[float] = 0.05

    def __init__(
        self,
        cfg: GripperVRRetargeterCfg,
    ):
        super().__init__(cfg)
        """Initialize the gripper retargeter."""
        # Store the hand to track
        if cfg.bound_hand not in [DeviceBase.TrackingTarget.CONTROLLER_LEFT, DeviceBase.TrackingTarget.CONTROLLER_RIGHT]:
            raise ValueError(
                "bound_hand must be either DeviceBase.TrackingTarget.CONTROLLER_LEFT or DeviceBase.TrackingTarget.CONTROLLER_RIGHT"
            )
        self.bound_hand = cfg.bound_hand

        # Initialize gripper in its open state
        self._previous_gripper_command = False

    def retarget(self, data: dict) -> torch.Tensor:
        """Convert hand joint poses to gripper command.

        Args:
            data: Dictionary with MotionControllerTrackingTarget.LEFT/RIGHT keys
                Each value is a 2D array: [pose(7), inputs(7)]

        Returns:
            torch.Tensor: Tensor containing a single bool value where True = close gripper, False = open gripper
        """
        # Extract controller data
        # controller_data = data[self.bound_hand]
        controller_data = data.get(self.bound_hand, np.array([]))

        # Calculate gripper command with hysteresis
        gripper_command_bool = self._calculate_gripper_command(controller_data, self._previous_gripper_command)
        gripper_value = -1.0 if gripper_command_bool else 1.0

        return torch.tensor([gripper_value], dtype=torch.float32, device=self._sim_device)

    def get_requirements(self) -> list[RetargeterBase.Requirement]:
        return [RetargeterBase.Requirement.MOTION_CONTROLLER]

    def _calculate_gripper_command(self, controller_data: np.ndarray, prev_state: float) -> bool:
        """Calculate gripper command from finger positions with hysteresis.

        Args:
            controller_data: 2D array [pose(7), inputs(7)]
            prev_state: Previous hand state (0.0 or 1.0)

        Returns:
            Hand state as bool (True = close, False = open)
        """
        if len(controller_data) <= DeviceBase.MotionControllerDataRowIndex.INPUTS.value:
            return False

        # Extract inputs from second row
        inputs = controller_data[DeviceBase.MotionControllerDataRowIndex.INPUTS.value]
        if len(inputs) < len(DeviceBase.MotionControllerInputIndex):
            return False

        # Extract specific inputs using enum
        trigger = inputs[DeviceBase.MotionControllerInputIndex.TRIGGER.value]  # 0.0 to 1.0 (analog)

        # Apply hysteresis to prevent rapid switching
        if trigger > self.GRIPPER_OPEN_METERS:
            self._previous_gripper_command = True
        elif trigger < self.GRIPPER_CLOSE_METERS:
            self._previous_gripper_command = False

        return self._previous_gripper_command


@dataclass
class GripperVRRetargeterCfg(RetargeterCfg):
    """Configuration for gripper retargeter."""

    bound_hand: DeviceBase.TrackingTarget = DeviceBase.TrackingTarget.CONTROLLER_RIGHT
    retargeter_type: type[RetargeterBase] = GripperVRRetargeter
