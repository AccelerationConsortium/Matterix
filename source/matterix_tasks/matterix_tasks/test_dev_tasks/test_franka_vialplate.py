# Copyright (c) 2022-2026, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dynamic Reference-Vial task using promoted Matterix labware.

The scene uses the promoted Fisherbrand 03-339-21F asset rather
than colour-coded cylinders. Every Vial is a gravity-enabled rigid body. Its
initial root pose is derived from the canonical holder-hole contract, with a
small clearance so PhysX—not a kinematic constraint—performs the final settle.

Set MATTERIX_VIAL_HOLDER_FRAME_DEBUG=1 to render the holder's physical opening
frames during a GUI or WebRTC review.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.sensors import OffsetCfg
from isaaclab.utils import configclass

from matterix.envs import mdp
from matterix_assets.labware.vials import (
    FISHERBRAND_03_339_21F_CFG,
    FISHERBRAND_03_339_21F_FRAME_OFFSETS,
    FISHERBRAND_03_339_21F_USD_PATH,
)
from matterix_assets.labware.vialplates import (
    VIALPLATE_3_5_CFG,
    VIALPLATE_3_5_FRAME_CONTRACT_PATH,
    VIALPLATE_3_5_USD_PATH,
)
from matterix.managers.semantics.primitive_semantics import IsInContactPhysicsCfg
from matterix_sm import MoveRelativeCfg, MoveToFrameCfg, OpenGripperCfg, PickObjectCfg
from matterix_sm.robot_action_spaces import FRANKA_IK_ACTION_SPACE

from .test_franka_beaker_lift import FrankaBeakerLiftEnvTestCfg


OFFICIAL_HOLDER_USD_PATH = Path(VIALPLATE_3_5_USD_PATH)
OFFICIAL_VIAL_USD_PATH = Path(FISHERBRAND_03_339_21F_USD_PATH)
OFFICIAL_FRAME_CONTRACT_PATH = Path(VIALPLATE_3_5_FRAME_CONTRACT_PATH)
FRAME_DEBUG_ENVIRONMENT_VARIABLE = "MATTERIX_VIAL_HOLDER_FRAME_DEBUG"
HOLDER_INITIAL_POS = (0.65, -0.18, -0.001)  # 2 mm above the measured -3 mm Seattle-table support surface.
VIAL_CONTACT_OFFSET_M = 1.0e-6
VIAL_CONTACT_FILTERS = [
    "robot/panda_leftfinger",
    "robot/panda_rightfinger",
    "vial_holder",
    "witness_vial_middle_mid_right",
    "witness_vial_middle_mid_left",
    "witness_vial_middle_left",
]
VIAL_INITIAL_SETTLING_CLEARANCE_M = 0.002
# The free-standing holder settles a fraction of a millimetre on the table
# before the vial reaches its floor. Keep the measured correction explicit.
VIAL_INITIAL_XY_MAPPING_M = (-0.00008, 0.00039)
# The holder/table contact stack needs the same 240 Hz physics cadence as the
# standalone qualification harness; 60 Hz produces measurable penetration and tilt.
VIALPLATE_PHYSICS_DT = 1.0 / 240.0
HOLDER_PLACE_FRAME_Z_M = 0.00835 + FISHERBRAND_03_339_21F_FRAME_OFFSETS["grasp"].pos[2]
HOLDER_PRE_PLACE_FRAME_Z_M = HOLDER_PLACE_FRAME_Z_M + 0.1
HOLDER_FRAME_IDENTITY_QUAT = (1.0, 0.0, 0.0, 0.0)
# The elevated view keeps the Franka/table context while making the holder
# and hole alignment legible.
REVIEW_VIEWER_EYE = (0.40, -0.05, 0.60)
REVIEW_VIEWER_LOOKAT = (0.65, -0.18, 0.04)


def _official_asset_paths() -> tuple[Path, Path, Path]:
    # Locate the promoted holder, vial, and canonical frame contract.
    holder_usd = OFFICIAL_HOLDER_USD_PATH
    vial_usd = OFFICIAL_VIAL_USD_PATH
    frame_contract = OFFICIAL_FRAME_CONTRACT_PATH
    missing = [str(path) for path in (holder_usd, vial_usd, frame_contract) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing promoted Matterix asset file(s): " + ", ".join(missing))
    return holder_usd, vial_usd, frame_contract


def _load_holder_hole_contract(path: Path) -> tuple[dict[str, OffsetCfg], str, tuple[str, ...], dict[str, object]]:
    """Return all physical hole frames and the single source of vial role selection."""
    contract = json.loads(path.read_text(encoding="utf-8"))
    frames = contract.get("frames")
    orientation = contract.get("frame_orientation_wxyz")
    selection = contract.get("initial_dynamic_vial_set")
    if not isinstance(frames, list) or len(frames) != 15:
        raise ValueError("holder-hole contract must declare exactly 15 frames")
    if orientation != [1.0, 0.0, 0.0, 0.0]:
        raise ValueError("holder-hole frames must be identity-oriented in holder coordinates")
    if not isinstance(selection, dict):
        raise ValueError("holder-hole contract is missing its Dynamic Vial Set")

    offsets: dict[str, OffsetCfg] = {}
    for frame in frames:
        if not isinstance(frame, dict):
            raise ValueError("holder-hole frame must be an object")
        name = frame.get("name")
        position = frame.get("position_m")
        if not isinstance(name, str) or not isinstance(position, list) or len(position) != 3:
            raise ValueError(f"malformed holder-hole frame: {frame!r}")
        if name in offsets:
            raise ValueError(f"duplicate holder-hole frame: {name}")
        offsets[name] = OffsetCfg(pos=tuple(float(value) for value in position), rot=tuple(orientation))

    pick = selection.get("pick_vial")
    witnesses = selection.get("witness_vials")
    if not isinstance(pick, str) or not isinstance(witnesses, list) or not all(isinstance(name, str) for name in witnesses):
        raise ValueError("holder-hole Dynamic Vial Set is malformed")
    if len(witnesses) != 3 or len(set((pick, *witnesses))) != 4:
        raise ValueError("holder-hole Dynamic Vial Set must contain one distinct pick and three witnesses")
    if pick not in offsets or not set(witnesses).issubset(offsets):
        raise ValueError("holder-hole Dynamic Vial Set names are absent from the frame grid")
    return offsets, pick, tuple(witnesses), contract


def _vial_object_name(hole_name: str, *, pick: bool) -> str:
    """Make role-bearing names without encoding any world-space coordinates."""
    if pick:
        return "pick_vial"
    return "witness_vial_" + hole_name.removeprefix("hole_")


def _dynamic_vial_layout(frame_contract_path: Path) -> tuple[dict[str, str], dict[str, tuple[float, float, float]]]:
    """Derive role names and initial Vial roots from holder opening frames.

    The root is intentionally two millimetres above its nominal 30 mm insertion
    depth. That clearance proves that the observed seating comes from gravity
    and holder collision, instead of an attached or kinematic presentation rig.
    """
    offsets, pick, witnesses, contract = _load_holder_hole_contract(frame_contract_path)
    heights = contract.get("reference_heights_m")
    if not isinstance(heights, dict):
        raise ValueError("holder-hole contract is missing reference heights")
    try:
        opening_plane = float(heights["opening_plane"])
        nominal_vial_bottom = float(heights["nominal_vial_bottom_at_30mm_insertion"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("holder-hole contract has malformed vial seating heights") from error
    if nominal_vial_bottom >= opening_plane:
        raise ValueError("nominal Vial bottom must be below the holder opening plane")

    role_to_hole = {_vial_object_name(pick, pick=True): pick}
    role_to_hole.update({_vial_object_name(hole, pick=False): hole for hole in witnesses})
    positions: dict[str, tuple[float, float, float]] = {}
    for role_name, hole_name in role_to_hole.items():
        opening = offsets[hole_name].pos
        positions[role_name] = (
            HOLDER_INITIAL_POS[0] + opening[0] + VIAL_INITIAL_XY_MAPPING_M[0],
            HOLDER_INITIAL_POS[1] + opening[1] + VIAL_INITIAL_XY_MAPPING_M[1],
            HOLDER_INITIAL_POS[2] + opening[2] - opening_plane + nominal_vial_bottom + VIAL_INITIAL_SETTLING_CLEARANCE_M,
        )
    return role_to_hole, positions


def _holder_placement_frame_offsets(hole_offsets: dict[str, OffsetCfg]) -> dict[str, OffsetCfg]:
    """Create robot grasping-frame targets above each physical holder opening."""
    placement_frames: dict[str, OffsetCfg] = {}
    for hole_name, hole_offset in hole_offsets.items():
        row_column = hole_name.removeprefix("hole_")
        x, y, _ = hole_offset.pos
        placement_frames[f"place_{row_column}"] = OffsetCfg(
            pos=(x, y, HOLDER_PLACE_FRAME_Z_M),
            rot=HOLDER_FRAME_IDENTITY_QUAT,
        )
        placement_frames[f"pre_place_{row_column}"] = OffsetCfg(
            pos=(x, y, HOLDER_PRE_PLACE_FRAME_Z_M),
            rot=HOLDER_FRAME_IDENTITY_QUAT,
        )
    return placement_frames


def _debug_frames_enabled() -> bool:
    return os.environ.get(FRAME_DEBUG_ENVIRONMENT_VARIABLE, "").strip().lower() in {"1", "true", "yes", "on"}


@configclass
class FreeStandingVialHolderCfg(VIALPLATE_3_5_CFG):
    """One dynamic holder with the canonical 15-hole FrameTransformer sensors."""

    frame_contract_path: str = ""
    debug_frame_vis: bool = False
    pick_vial_hole_name: str = ""
    witness_vial_hole_names: tuple[str, ...] = ()

    def __post_init__(self):
        offsets, pick, witnesses, _ = _load_holder_hole_contract(Path(self.frame_contract_path))
        # MatterixRigidObjectCfg turns every offset into a FrameTransformer
        # named {frame_name}_{asset_name}; keep the physical hole API and add
        # grasping-frame placement targets as separate, explicit names.
        self.frames = {**offsets, **_holder_placement_frame_offsets(offsets)}
        self.sensors = {}
        self.pick_vial_hole_name = pick
        self.witness_vial_hole_names = witnesses
        super().__post_init__()
        self.spawn.rigid_props = sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=False,
            disable_gravity=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=2,
            sleep_threshold=0.01,
            stabilization_threshold=0.01,
            max_depenetration_velocity=1.0,
            linear_damping=0.05,
            angular_damping=0.1,
        )
        for frame_name, sensor_cfg in self.sensors.items():
            sensor_cfg.debug_vis = self.debug_frame_vis
            # Isaac Lab's default frame marker is intentionally large (30 mm in
            # MatterixRigidObjectCfg). That is useful for a robot workspace,
            # but it is larger than the 21 mm vial bore, so the arrow tips look
            # displaced from a correctly centred hole. Give this diagnostic
            # task a private, compact marker config and an explicit path per
            # hole. The frame origin remains the FrameTransformer target;
            # only the inspection glyph is being changed.
            visualizer_cfg = copy.deepcopy(sensor_cfg.visualizer_cfg)
            visualizer_cfg.prim_path = f"/Visuals/VialHolderFrames/{frame_name}"
            visualizer_cfg.markers["frame"].scale = (0.012, 0.012, 0.012)
            visualizer_cfg.markers["connecting_line"].radius = 0.0004
            sensor_cfg.visualizer_cfg = visualizer_cfg


@configclass
class ReferenceVialCfg(FISHERBRAND_03_339_21F_CFG):
    """Dynamic closed Reference Vial with Franka cap-grasp frames."""

    def __post_init__(self):
        self.frames = {**self.frames, **FISHERBRAND_03_339_21F_FRAME_OFFSETS}
        self.sensors = {}
        self.semantics = [IsInContactPhysicsCfg(filter_prim_paths_expr=VIAL_CONTACT_FILTERS)]
        super().__post_init__()
        # Match the staged-candidate drop test: dynamic gravity, conservative
        # contact offsets, high enough solver iterations, and damping for the
        # narrow 0.25 mm radial holder clearance. Nothing is kinematic.
        self.spawn = sim_utils.UsdFileCfg(
            usd_path=self.usd_path,
            mass_props=sim_utils.MassPropertiesCfg(mass=self.mass),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
                sleep_threshold=0.1,
                stabilization_threshold=0.1,
                max_depenetration_velocity=1.0,
                linear_damping=0.5,
                angular_damping=0.5,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=VIAL_CONTACT_OFFSET_M,
                rest_offset=0.0,
            ),
            scale=self.scale,
            activate_contact_sensors=True,
        )


@configclass
class VialplateObservationManagerCfg:
    """Robot state plus holder/Vial state required by the workflow."""

    @configclass
    class ArticulationsGroup(ObsGroup):
        robot__root_world_pos = ObsTerm(func=mdp.root_world_pos, params={"asset_name": "robot"})
        robot__root_world_quat = ObsTerm(func=mdp.root_world_quat, params={"asset_name": "robot"})
        robot__joint_pos = ObsTerm(func=mdp.joint_pos, params={"asset_name": "robot"})
        robot__joint_vel = ObsTerm(func=mdp.joint_vel, params={"asset_name": "robot"})
        robot__ee_world_pos = ObsTerm(func=mdp.ee_world_pos, params={"asset_name": "robot"})
        robot__ee_world_quat = ObsTerm(func=mdp.ee_world_quat, params={"asset_name": "robot"})
        robot__gripper_pos = ObsTerm(func=mdp.gripper_pos, params={"asset_name": "robot"})
        robot__grasping_frame_world_pos = ObsTerm(
            func=mdp.frame_world_pos, params={"asset_name": "robot", "frame_name": "grasping_frame"}
        )
        robot__grasping_frame_world_quat = ObsTerm(
            func=mdp.frame_world_quat, params={"asset_name": "robot", "frame_name": "grasping_frame"}
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class RigidObjectsGroup(ObsGroup):
        vial_holder__object_world_pos = ObsTerm(func=mdp.object_world_pos, params={"asset_name": "vial_holder"})
        vial_holder__object_world_quat = ObsTerm(func=mdp.object_world_quat, params={"asset_name": "vial_holder"})
        vial_holder__hole_middle_center_frame = ObsTerm(
            func=mdp.frame_world_pose, params={"asset_name": "vial_holder", "frame_name": "hole_middle_center"}
        )
        vial_holder__pre_place_middle_center_frame = ObsTerm(
            func=mdp.frame_world_pose, params={"asset_name": "vial_holder", "frame_name": "pre_place_middle_center"}
        )
        vial_holder__place_middle_center_frame = ObsTerm(
            func=mdp.frame_world_pose, params={"asset_name": "vial_holder", "frame_name": "place_middle_center"}
        )
        vial_holder__object_lin_vel = ObsTerm(func=mdp.object_lin_vel, params={"asset_name": "vial_holder"})
        vial_holder__object_ang_vel = ObsTerm(func=mdp.object_ang_vel, params={"asset_name": "vial_holder"})
        pick_vial__object_world_pos = ObsTerm(func=mdp.object_world_pos, params={"asset_name": "pick_vial"})
        pick_vial__object_world_quat = ObsTerm(func=mdp.object_world_quat, params={"asset_name": "pick_vial"})
        pick_vial__pre_grasp_frame = ObsTerm(
            func=mdp.frame_world_pose, params={"asset_name": "pick_vial", "frame_name": "pre_grasp"}
        )
        pick_vial__grasp_frame = ObsTerm(
            func=mdp.frame_world_pose, params={"asset_name": "pick_vial", "frame_name": "grasp"}
        )
        pick_vial__post_grasp_frame = ObsTerm(
            func=mdp.frame_world_pose, params={"asset_name": "pick_vial", "frame_name": "post_grasp"}
        )
        pick_vial__object_lin_vel = ObsTerm(func=mdp.object_lin_vel, params={"asset_name": "pick_vial"})
        witness_vial_middle_mid_right__object_world_pos = ObsTerm(
            func=mdp.object_world_pos, params={"asset_name": "witness_vial_middle_mid_right"}
        )
        witness_vial_middle_mid_left__object_world_pos = ObsTerm(
            func=mdp.object_world_pos, params={"asset_name": "witness_vial_middle_mid_left"}
        )
        witness_vial_middle_left__object_world_pos = ObsTerm(
            func=mdp.object_world_pos, params={"asset_name": "witness_vial_middle_left"}
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    articulations: ArticulationsGroup = ArticulationsGroup()
    rigid_objects: RigidObjectsGroup = RigidObjectsGroup()


@configclass
class FrankaVialplateEnvTestCfg(FrankaBeakerLiftEnvTestCfg):
    """Franka/table scene with one dynamic holder and four real dynamic Vials."""

    pick_vial_hole_name: str = ""
    witness_vial_hole_names: tuple[str, ...] = ()
    vial_object_to_hole: dict[str, str] = {}
    vial_initial_positions: dict[str, tuple[float, float, float]] = {}

    def __post_init__(self):
        super().__post_init__()
        # The inherited Franka IK action controls a virtual frame 107 mm from
        # panda_hand, while the shared diagnostic sensors use 103.4 mm. Keep
        # this task observed frame aligned with the actual IK target so the
        # 1 mm manipulation gates measure the commanded frame rather than a
        # 3.6 mm bookkeeping offset.
        self.articulated_assets = dict(self.articulated_assets)
        robot = copy.deepcopy(self.articulated_assets["robot"])
        robot.sensors["ee_frame"].target_frames[0].offset = OffsetCfg(pos=(0.0, 0.0, 0.107))
        robot.sensors["grasping_frame"].target_frames[0].offset = OffsetCfg(
            pos=(0.0, 0.0, 0.107), rot=(0.0, 1.0, 0.0, 0.0)
        )
        self.articulated_assets["robot"] = robot
        self.sim.dt = VIALPLATE_PHYSICS_DT
        _, _, frame_contract = _official_asset_paths()
        holder = FreeStandingVialHolderCfg(
            frame_contract_path=str(frame_contract),
            pos=HOLDER_INITIAL_POS,
            activate_contact_sensors=True,
            debug_frame_vis=_debug_frames_enabled(),
            semantic_tags=[("class", "vial_holder")],
        )
        vial_object_to_hole, vial_initial_positions = _dynamic_vial_layout(frame_contract)

        self.objects = dict(self.objects)
        self.objects.pop("beaker", None)
        self.objects["vial_holder"] = holder
        for vial_name, initial_pos in vial_initial_positions.items():
            self.objects[vial_name] = ReferenceVialCfg(
                pos=initial_pos,
                activate_contact_sensors=True,
                semantic_tags=[("class", "vial")],
            )

        # The inherited beaker-only reset and workflow are replaced by the
        # dynamic four-Vial pick-and-return scene.
        self.events.randomize_beaker_position = None
        # This qualification task is driven by headless workflow diagnostics;
        # do not contend with the parent beaker test recorder path.
        self.record_path = None
        self.observations = VialplateObservationManagerCfg()
        self.pick_vial_hole_name = holder.pick_vial_hole_name
        self.witness_vial_hole_names = holder.witness_vial_hole_names
        self.vial_object_to_hole = vial_object_to_hole
        self.vial_initial_positions = vial_initial_positions
        pick_hole_suffix = self.pick_vial_hole_name.removeprefix("hole_")
        # Provisional evidence route: incremental insertion is used because the
        # nominal one-jump PlaceObject route times out against the measured
        # 0.25 mm dynamic vial/well clearance. Keep this route until the
        # clearance and collision policy are confirmed; it is not Ticket 10
        # acceptance by itself.
        pick_action = PickObjectCfg(
            description="Pick the Reference Vial by its cap",
            agent_assets="robot",
            object="pick_vial",
            post_grasp_offset=(0.0, 0.0, 0.02),
            action_space_info=FRANKA_IK_ACTION_SPACE,
        )
        staged_lift_actions = [
            MoveRelativeCfg(
                agent_assets="robot",
                position_offset=(0.0, 0.0, 0.02),
                orientation_offset=None,
                position_threshold=0.001,
                orientation_threshold=0.02,
                settling_time=0.05,
                action_space_info=FRANKA_IK_ACTION_SPACE,
            )
            for _ in range(4)
        ]
        pre_place_action = MoveToFrameCfg(
            object="vial_holder",
            frame=f"pre_place_{pick_hole_suffix}",
            agent_assets="robot",
            position_threshold=0.001,
            orientation_threshold=0.02,
            settling_time=0.05,
            use_frame_orientation=False,
            action_space_info=FRANKA_IK_ACTION_SPACE,
        )
        insertion_actions = [
            MoveRelativeCfg(
                agent_assets="robot",
                position_offset=(0.0, 0.0, -0.02),
                orientation_offset=None,
                position_threshold=0.001,
                orientation_threshold=0.02,
                settling_time=0.05,
                action_space_info=FRANKA_IK_ACTION_SPACE,
            )
            for _ in range(5)
        ]
        release_action = OpenGripperCfg(
            agent_assets="robot",
            action_space_info=FRANKA_IK_ACTION_SPACE,
        )
        retreat_action = MoveRelativeCfg(
            agent_assets="robot",
            position_offset=(0.0, 0.0, 0.1),
            orientation_offset=None,
            position_threshold=0.01,
            orientation_threshold=0.02,
            settling_time=0.05,
            action_space_info=FRANKA_IK_ACTION_SPACE,
        )
        self.workflows = {
            "pick_return_vial": [
                pick_action,
                *staged_lift_actions,
                pre_place_action,
                *insertion_actions,
                release_action,
                retreat_action,
            ]
        }
        self.viewer.eye = REVIEW_VIEWER_EYE
        self.viewer.lookat = REVIEW_VIEWER_LOOKAT
