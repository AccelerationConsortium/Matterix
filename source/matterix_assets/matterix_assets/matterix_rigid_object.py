# Copyright (c) 2022-2025, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for a rigid object."""

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers.event_manager import EventTermCfg
from isaaclab.sensors import FrameTransformerCfg, OffsetCfg
from isaaclab.utils import configclass

from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip

##
# Configuration
##

marker_cfg = FRAME_MARKER_CFG.copy()
marker_cfg.markers["frame"].scale = (0.03, 0.03, 0.03)
marker_cfg.prim_path = "/Visuals/FrameTransformer"


@configclass
class MatterixRigidObjectCfg(RigidObjectCfg):
    """Configuration parameters for a rigid object."""

    # defined by the user in the scene description
    pos: tuple[float, float, float] | None = None  # default value is (0.0, 0.0, 0.0)
    # Quaternion rotation (w, x, y, z)
    rot: tuple[float, float, float, float] | None = None  # default value is (1.0, 0.0, 0.0, 0.0)

    # defined by the user for each rigid object
    prim_path = "{ENV_REGEX_NS}/RigidObjects"

    usd_path: str = MISSING
    scale: tuple[float, float, float] | None = None
    mass: float | None = None
    activate_contact_sensors: bool = False

    event_terms: dict[str, EventTermCfg] = {}

    frames: dict[str, tuple[float, float, float] | OffsetCfg] = {}  # it will be converted to sensors in __post_init__
    sensors: dict[str, FrameTransformerCfg] = {}

    semantic_tags: list[tuple] = []

    def __post_init__(self):
        if hasattr(super(), "__post_init__"):
            super().__post_init__()
        # spawn configuration
        self.spawn = sim_utils.UsdFileCfg(
            usd_path=self.usd_path,
            mass_props=sim_utils.MassPropertiesCfg(mass=self.mass),  # kg
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            scale=self.scale,
            activate_contact_sensors=self.activate_contact_sensors,
        )

        # initial state configuration
        if self.pos is not None:
            self.init_state.pos = self.pos
        if self.rot is not None:
            self.init_state.rot = self.rot

        # map the frames to sensors as this is supported by the framework
        for frame_name, frame_val in self.frames.items():
            if isinstance(frame_val, OffsetCfg):
                # If frame_val is already an OffsetCfg, use it directly
                offset = frame_val
            else:
                # Otherwise, create a new OffsetCfg with the position
                offset = OffsetCfg(pos=frame_val)

            self.sensors[frame_name] = FrameTransformerCfg(
                prim_path="",
                debug_vis=False,
                visualizer_cfg=marker_cfg,
                target_frames=[
                    FrameTransformerCfg.FrameCfg(
                        prim_path="",
                        name=frame_name,
                        offset=offset,
                    ),
                ],
            )
