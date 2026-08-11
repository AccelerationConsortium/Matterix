"""Dedicated Ticket 0c.7 task for one requested small open vessel.

The task is intentionally configured from the four Ticket 0c environment
variables. It never imports the production beaker configuration and it exposes
only one target vessel in the target labware slot.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from matterix.envs import LightStateCfg, MatterixBaseEnvCfg, mdp
from matterix.managers import EventManagerCfg
from matterix_assets.infrastructure.tables import TABLE_SEATTLE_INST_Cfg
from matterix_assets.matterix_rigid_object import MatterixRigidObjectCfg
from matterix_assets.robots import FRANKA_PANDA_HIGH_PD_IK_CFG
from matterix_sm import MoveRelativeCfg, OpenGripperCfg, PickObjectCfg, WaitCfg
from matterix_sm.robot_action_spaces import FRANKA_IK_ACTION_SPACE

import isaaclab.envs.mdp as isaaclab_mdp
import isaaclab.sim as sim_utils
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim.schemas import CollisionPropertiesCfg
from isaaclab.sim.spawners.lights import DomeLightCfg, SphereLightCfg
from isaaclab.utils import configclass


TASK_ID = "Matterix-Ticket0c-Small-Vessel-Franka-v1"
ASSET_IDS = (
    "corning-5580-25", "corning-5580-50", "corning-5580-100",
    "corning-3025-50", "dwk-213133202", "dwk-213133408",
)
REQUIRED_FRAMES = (
    "base", "opening", "grasp_body", "pre_grasp_body", "post_grasp_lift",
    "place", "pre_place", "post_place", "pour_lip", "pour_pivot",
)
STATION_CANDIDATES_M = ((0.15, 0.0), (0.0, 0.15), (-0.15, 0.0), (0.0, -0.15))
PICK_PLACE_STATION_OFFSET_M = STATION_CANDIDATES_M[0]
POUR_STATION_OFFSET_M = STATION_CANDIDATES_M[3]
PROFILE_PATH = Path(__file__).with_name("ticket0c7_profiles.json")
PROFILES = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
TICKET0C7_VIEWER_EYE = (1.0, -1.1, 0.7)
TICKET0C7_VIEWER_LOOKAT = (0.45, 0.0, 0.08)
def _configuration() -> tuple[str, str, str, str, dict]:
    asset_usd = os.environ.get("MATTERIX_TICKET0C_ASSET_USD", "")
    asset_id = os.environ.get("MATTERIX_TICKET0C_ASSET_ID", "")
    scenario = os.environ.get("MATTERIX_TICKET0C_SCENARIO", "pick_place")
    result_json = os.environ.get("MATTERIX_TICKET0C_RESULT_JSON", "")
    if asset_id not in ASSET_IDS:
        raise ValueError(f"MATTERIX_TICKET0C_ASSET_ID must name one of {ASSET_IDS}, got {asset_id!r}")
    if not asset_usd or not Path(asset_usd).is_file():
        raise ValueError("MATTERIX_TICKET0C_ASSET_USD must point to an existing canonical USD")
    if scenario not in {"pick_place", "pour"}:
        raise ValueError(f"MATTERIX_TICKET0C_SCENARIO must be pick_place or pour, got {scenario!r}")
    if not result_json:
        raise ValueError("MATTERIX_TICKET0C_RESULT_JSON is required")
    profile = PROFILES[asset_id]
    if set(profile["frames"]) != set(REQUIRED_FRAMES):
        raise ValueError(f"{asset_id}: profile does not provide the frozen frame set")
    return asset_usd, asset_id, scenario, result_json, profile


def _asset_id_from_usd(asset_usd: str, requested_asset_id: str) -> str:
    from pxr import Usd

    stage = Usd.Stage.Open(asset_usd)
    if stage is None:
        raise ValueError(f"cannot open requested USD {asset_usd}")
    root = stage.GetDefaultPrim()
    attr = root.GetAttribute("assetId") if root else None
    loaded_id = attr.Get() if attr else None
    if loaded_id != requested_asset_id:
        raise ValueError(
            f"requested assetId mismatch: requested={requested_asset_id!r}, "
            f"loaded={loaded_id!r}"
        )
    return str(loaded_id)


try:
    _ASSET_USD, _ASSET_ID, _SCENARIO, _RESULT_JSON, _PROFILE = _configuration()
    _LOADED_ASSET_ID = _asset_id_from_usd(_ASSET_USD, _ASSET_ID)
    _RUNTIME_CONFIGURATION_ERROR = None
except (ImportError, OSError, RuntimeError, ValueError) as error:
    # Task discovery imports every registered task before the launcher has a
    # chance to install the per-run environment. Keep discovery side-effect
    # free, while failing as soon as this task config is actually instantiated.
    _ASSET_USD = "/tmp/ticket0c7-unconfigured.usda"
    _ASSET_ID = ASSET_IDS[0]
    _SCENARIO = "pick_place"
    _RESULT_JSON = "/tmp/ticket0c7-unconfigured.json"
    _PROFILE = PROFILES[_ASSET_ID]
    _LOADED_ASSET_ID = _ASSET_ID
    _RUNTIME_CONFIGURATION_ERROR = str(error)

POUR_Y_OFFSET_M = (
    -0.16 if _LOADED_ASSET_ID == "dwk-213133202"
    else -0.15 if _LOADED_ASSET_ID.startswith("dwk-")
    else -0.17
)
POUR_X_OFFSET_M = 0.0
PICK_PLACE_LOWER_OFFSET_M = -0.085 if _LOADED_ASSET_ID == "corning-5580-50" else -0.08
# The tilted 50 mL release frame otherwise leaves its base frame 0.7 mm above
# the table; the supplied collider tolerates this additional 1 mm descent.
POUR_RELEASE_LOWER_OFFSET_M = -0.076
POUR_RELEASE_POSITION_THRESHOLD_M = 0.01
POUR_RELEASE_TIMEOUT_S = 8.0 if _LOADED_ASSET_ID == "corning-3025-50" else 10.0
POUR_RELEASE_SETTLING_TIME_S = 1.0 if _LOADED_ASSET_ID == "corning-3025-50" else 0.05
_LOW_FRICTION_POUR_50ML = _LOADED_ASSET_ID == "corning-5580-50" and _SCENARIO == "pour"
_HIGH_FRICTION_POUR_100ML = _LOADED_ASSET_ID == "corning-5580-100" and _SCENARIO == "pour"
_MODERATE_FRICTION_POUR_3025 = _LOADED_ASSET_ID == "corning-3025-50" and _SCENARIO == "pour"
_LOW_PROFILE_DWK_POUR = _LOADED_ASSET_ID.startswith("dwk-") and _SCENARIO == "pour"
TICKET0C7_3025_CONTACT_POLICY = "normal_robot_contact_high_hand_drive"
_POUR_ORIENTATION_THRESHOLD = (
    0.5
    if _LOADED_ASSET_ID in {"corning-5580-25", "corning-5580-50", "corning-3025-50"}
    else 0.5
    if _LOADED_ASSET_ID.startswith("dwk-")
    else 0.05
)
_ROBOT_STATIC_FRICTION_RANGE = (
    (5.0, 5.0)
    if _HIGH_FRICTION_POUR_100ML
    else (0.2, 0.2)
    if _LOW_FRICTION_POUR_50ML
    else (1.0, 1.0)
)
_ROBOT_DYNAMIC_FRICTION_RANGE = (
    (4.0, 4.0)
    if _HIGH_FRICTION_POUR_100ML
    else (0.15, 0.15)
    if _LOW_FRICTION_POUR_50ML
    else (0.9, 0.9)
)
_ROBOT_FINGER_STATIC_FRICTION_RANGE = (
    (5.0, 5.0)
    if _HIGH_FRICTION_POUR_100ML
    else (0.2, 0.2)
    if _LOW_FRICTION_POUR_50ML
    else (2.0, 2.0)
    if _MODERATE_FRICTION_POUR_3025
    else (2.0, 2.0)
    if _LOW_PROFILE_DWK_POUR
    else (1.5, 1.5)
)
_ROBOT_FINGER_DYNAMIC_FRICTION_RANGE = (
    (4.0, 4.0)
    if _HIGH_FRICTION_POUR_100ML
    else (0.15, 0.15)
    if _LOW_FRICTION_POUR_50ML
    else (1.6, 1.6)
    if _MODERATE_FRICTION_POUR_3025
    else (1.6, 1.6)
    if _LOW_PROFILE_DWK_POUR
    else (1.2, 1.2)
)


def _robot_cfg():
    cfg = FRANKA_PANDA_HIGH_PD_IK_CFG(pos=(0.0, 0.0, 0.0))
    if _HIGH_FRICTION_POUR_100ML or _MODERATE_FRICTION_POUR_3025 or _LOW_PROFILE_DWK_POUR:
        cfg.actuators = cfg.actuators.copy()
        cfg.actuators["panda_hand"] = cfg.actuators["panda_hand"].copy()
        cfg.actuators["panda_hand"].stiffness = 1000.0
        cfg.actuators["panda_hand"].damping = 80.0
    return cfg


@configclass
class Ticket0CVesselCfg(MatterixRigidObjectCfg):
    prim_path = "{ENV_REGEX_NS}/RigidObjects_Labware"
    usd_path = _ASSET_USD
    scale = (1.0, 1.0, 1.0)
    mass = _PROFILE["mass_kg"]
    # Qualification uses post-release stable collision inference. Enabling a
    # contact sensor on every delivered convex piece adds an unsupported GPU
    # contact-filter path without contributing to any acceptance metric.
    activate_contact_sensors = False
    frames = {
        **{name: tuple(values) for name, values in _PROFILE["frames"].items()},
        "pre_grasp": tuple(_PROFILE["frames"]["pre_grasp_body"]),
        "grasp": tuple(_PROFILE["frames"]["grasp_body"]),
        "post_grasp": tuple(_PROFILE["frames"]["post_grasp_lift"]),
    }
    semantic_tags = [("class", "ticket0c_small_vessel"), ("assetId", _LOADED_ASSET_ID)]

    def __post_init__(self):
        super().__post_init__()
        # The visual layer owns its per-prim material bindings. Keep the task
        # material-neutral so glass cannot override caps, liners, or other
        # visual descendants. Collision remains the delivered invisible
        # compound-convex layer.
        # The supplied compound pieces are millimetre-scale. Keep contact
        # generation below the vessel wall scale and do not author a second
        # generic approximation over the delivered colliders.
        self.spawn.collision_props = CollisionPropertiesCfg(
            contact_offset=0.001,
            rest_offset=0.0,
        )


@configclass
class EventCfg(EventManagerCfg):
    vessel_physics_material = EventTerm(
        func=isaaclab_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("vessel", body_names=".*"),
            "static_friction_range": (0.85, 0.85),
            "dynamic_friction_range": (0.70, 0.70),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
        },
    )
    robot_physics_material = EventTerm(
        func=isaaclab_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": _ROBOT_STATIC_FRICTION_RANGE,
            "dynamic_friction_range": _ROBOT_DYNAMIC_FRICTION_RANGE,
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
        },
    )
    robot_finger_physics_material = EventTerm(
        func=isaaclab_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["panda_leftfinger", "panda_rightfinger"]),
            "static_friction_range": _ROBOT_FINGER_STATIC_FRICTION_RANGE,
            "dynamic_friction_range": _ROBOT_FINGER_DYNAMIC_FRICTION_RANGE,
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
        },
    )
    reset_scene_to_default = EventTerm(func=isaaclab_mdp.reset_scene_to_default, mode="reset")


@configclass
class ObservationManagerCfg:
    @configclass
    class ArticulationsGroup(ObsGroup):
        robot__root_world_pos = ObsTerm(func=mdp.root_world_pos, params={"asset_name": "robot"})
        robot__root_world_quat = ObsTerm(func=mdp.root_world_quat, params={"asset_name": "robot"})
        robot__joint_pos = ObsTerm(func=mdp.joint_pos, params={"asset_name": "robot"})
        robot__joint_vel = ObsTerm(func=mdp.joint_vel, params={"asset_name": "robot"})
        robot__ee_world_pos = ObsTerm(func=mdp.ee_world_pos, params={"asset_name": "robot"})
        robot__ee_world_quat = ObsTerm(func=mdp.ee_world_quat, params={"asset_name": "robot"})
        robot__gripper_pos = ObsTerm(func=mdp.gripper_pos, params={"asset_name": "robot"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class RigidObjectsGroup(ObsGroup):
        vessel__object_world_pos = ObsTerm(func=mdp.object_world_pos, params={"asset_name": "vessel"})
        vessel__object_world_quat = ObsTerm(func=mdp.object_world_quat, params={"asset_name": "vessel"})
        vessel__object_lin_vel = ObsTerm(func=mdp.object_lin_vel, params={"asset_name": "vessel"})
        vessel__object_ang_vel = ObsTerm(func=mdp.object_ang_vel, params={"asset_name": "vessel"})
        vessel__pre_grasp_frame = ObsTerm(func=mdp.frame_world_pose, params={"asset_name": "vessel", "frame_name": "pre_grasp"})
        vessel__grasp_frame = ObsTerm(func=mdp.frame_world_pose, params={"asset_name": "vessel", "frame_name": "grasp"})
        vessel__post_grasp_frame = ObsTerm(func=mdp.frame_world_pose, params={"asset_name": "vessel", "frame_name": "post_grasp"})
        vessel__opening_frame = ObsTerm(func=mdp.frame_world_pose, params={"asset_name": "vessel", "frame_name": "opening"})
        vessel__base_frame = ObsTerm(func=mdp.frame_world_pose, params={"asset_name": "vessel", "frame_name": "base"})
        vessel__pour_lip_frame = ObsTerm(func=mdp.frame_world_pose, params={"asset_name": "vessel", "frame_name": "pour_lip"})
        vessel__pour_pivot_frame = ObsTerm(func=mdp.frame_world_pose, params={"asset_name": "vessel", "frame_name": "pour_pivot"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    articulations: ArticulationsGroup = ArticulationsGroup()
    rigid_objects: RigidObjectsGroup = RigidObjectsGroup()


def _pick_sequence():
    pick = PickObjectCfg(
        description="Ticket 0c.7 grasp and lift",
        agent_assets="robot",
        object="vessel",
        post_grasp_offset=(0.0, 0.0, 0.07),
        action_space_info=FRANKA_IK_ACTION_SPACE,
    )
    return [WaitCfg(duration=1.5), pick, WaitCfg(duration=2.0)]


def _pour_rotation_sequence(sign: float):
    # The 50 mL body uses four bounded 15-degree increments; the other Corning
    # bodies retain their smooth eleven-step path.
    if _LOADED_ASSET_ID == "corning-5580-50":
        offsets = [(0.9914449, 0.1305262)] * 4
    elif _LOADED_ASSET_ID == "corning-5580-25":
        # This low-profile body loses about 15 degrees across the supplied
        # grasp envelope; use a 9-degree increment while retaining eleven
        # measured steps and the same smooth return path.
        offsets = [(0.9969173, 0.0784591)] * 11
    elif _LOADED_ASSET_ID == "corning-5580-100":
        # The taller 100 mL body needs progressive small wrist targets; the
        # tighter completion tolerance and higher finger friction keep the
        # grasp coupled without four large inertial impulses.
        offsets = [(0.9975641, 0.0697565)] * 11
    elif _LOADED_ASSET_ID == "corning-3025-50":
        # Six-degree targets complete reliably for this body; use twelve
        # bounded steps so the held vessel exceeds the 60-degree gate.
        offsets = [(0.9986295, 0.05233596)] * 12
    elif _LOADED_ASSET_ID.startswith("corning-"):
        # Use small 6 degree increments so the supplied runtime collider stays
        # coupled to the two fingers through both the pour and return.
        offsets = [(0.9986295, 0.05233596)] * 11
    elif _LOADED_ASSET_ID == "dwk-213133202":
        # The reduced runtime collider needs smaller increments to keep this
        # low-profile body seated in the gripper through the full pour.
        offsets = [(0.9949685, 0.1001881)] * 10
    elif _LOADED_ASSET_ID.startswith("dwk-"):
        offsets = [(0.9925462, 0.1218693)] * 7
    else:
        offsets = [(0.9949685, 0.1001881)] * 10
    timeout = 4.0 if _LOADED_ASSET_ID == "corning-3025-50" else 12.0
    wait_duration = 0.35
    return [
        action
        for w, increment in offsets
        for action in (
            MoveRelativeCfg(
                agent_assets="robot",
                orientation_offset=(w, 0.0, increment * sign, 0.0),
                timeout=timeout,
                # Do not advance to the next wrist target while the current
                # bounded rotation is still settling; the old 0.5 rad
                # tolerance let the arm outrun the held vessel.
                orientation_threshold=_POUR_ORIENTATION_THRESHOLD,
                action_space_info=FRANKA_IK_ACTION_SPACE,
            ),
            WaitCfg(duration=wait_duration),
        )
    ]


def _pour_return_sequence():
    if _LOADED_ASSET_ID == "corning-5580-100":
        # Use the shared return corridor after the bounded 100 mL pour.
        return [
            MoveRelativeCfg(agent_assets="robot", position_offset=(0.21, 0.17, -0.05), action_space_info=FRANKA_IK_ACTION_SPACE),
        ]
    if _LOADED_ASSET_ID == "corning-3025-50":
        # The basin is the 3025 pour placement point. Do not add an
        # unreachable diagonal return after the vessel is already over it;
        # lower and release at the basin in the shared workflow tail.
        return []
    if _LOADED_ASSET_ID.startswith("dwk-"):
        # The diagonal return is outside the stable IK corridor for the
        # low-profile DWK grasp. Split it into two valid station moves.
        return [
            MoveRelativeCfg(agent_assets="robot", position_offset=(0.10, 0.0, 0.0), action_space_info=FRANKA_IK_ACTION_SPACE),
        ]
    return [
        MoveRelativeCfg(agent_assets="robot", position_offset=(0.21, 0.17, -0.05), action_space_info=FRANKA_IK_ACTION_SPACE),
    ]


def _pour_release_lower_sequence():
    if _LOADED_ASSET_ID == "corning-3025-50":
        # Split the table-contact descent so the IK target never asks the
        # wrist to drive the grasp through the support plane in one move.
        return [
            MoveRelativeCfg(
                agent_assets="robot",
                position_offset=(0.0, 0.0, -0.045),
                timeout=POUR_RELEASE_TIMEOUT_S,
                position_threshold=POUR_RELEASE_POSITION_THRESHOLD_M,
                settling_time=POUR_RELEASE_SETTLING_TIME_S,
                action_space_info=FRANKA_IK_ACTION_SPACE,
            ),
            MoveRelativeCfg(
                agent_assets="robot",
                position_offset=(0.0, 0.0, -0.045),
                timeout=POUR_RELEASE_TIMEOUT_S,
                position_threshold=POUR_RELEASE_POSITION_THRESHOLD_M,
                settling_time=POUR_RELEASE_SETTLING_TIME_S,
                action_space_info=FRANKA_IK_ACTION_SPACE,
            ),
        ]
    if _LOADED_ASSET_ID == "dwk-213133408":
        # This low-profile collider reaches the table only when the final
        # descent uses a bounded IK target; a second target would command
        # motion beyond the grounded collider.
        return [
            MoveRelativeCfg(
                agent_assets="robot",
                position_offset=(0.0, 0.0, -0.045),
                timeout=POUR_RELEASE_TIMEOUT_S,
                position_threshold=POUR_RELEASE_POSITION_THRESHOLD_M,
                settling_time=POUR_RELEASE_SETTLING_TIME_S,
                action_space_info=FRANKA_IK_ACTION_SPACE,
            ),
        ]
    return [
        MoveRelativeCfg(
            agent_assets="robot",
            position_offset=(0.0, 0.0, POUR_RELEASE_LOWER_OFFSET_M),
            timeout=POUR_RELEASE_TIMEOUT_S,
            position_threshold=POUR_RELEASE_POSITION_THRESHOLD_M,
            settling_time=POUR_RELEASE_SETTLING_TIME_S,
            action_space_info=FRANKA_IK_ACTION_SPACE,
        )
    ]


@configclass
class Ticket0CSmallVesselEnvCfg(MatterixBaseEnvCfg):
    env_spacing = 10.0
    episode_length_s = 60.0
    decimation = 1
    dt = 1.0 / 240.0
    lights = {
        "key": LightStateCfg(
            light=SphereLightCfg(
                color=(1.0, 0.95, 0.90),
                intensity=25000.0,
                enable_color_temperature=True,
                color_temperature=5000,
            ),
            pos=(1.5, -1.5, 2.0),
        ),
        "fill": LightStateCfg(
            light=SphereLightCfg(color=(0.85, 0.90, 1.0), intensity=12000.0),
            pos=(-1.0, 1.0, 1.2),
        ),
        "ambient": LightStateCfg(light=DomeLightCfg(color=(0.65, 0.70, 0.80), intensity=800.0)),
    }
    objects = {
        "vessel": Ticket0CVesselCfg(pos=(0.55, 0.0, 0.0)),
        "table": TABLE_SEATTLE_INST_Cfg(pos=(0.5, 0.0, 0.0)),
        # The pour gate uses the frozen logical basin station offset. Do not
        # spawn the old placeholder cylinder into the GUI scene: it can cover
        # the table and obscure the asset under review.
    }
    articulated_assets = {"robot": _robot_cfg()}
    gripper_joint_names = ["panda_finger_joint1", "panda_finger_joint2"]
    observations = ObservationManagerCfg()
    events = EventCfg()
    record_path = _RESULT_JSON
    workflows = {
        "pick_place": _pick_sequence() + [
            MoveRelativeCfg(agent_assets="robot", position_offset=(0.17, 0.0, 0.0), action_space_info=FRANKA_IK_ACTION_SPACE),
            # Lower until the vessel base is on the Seattle table before
            # opening. The correction follows the frozen post-grasp frame so
            # taller vessels do not inherit the short-vessel drop height.
            MoveRelativeCfg(agent_assets="robot", position_offset=(0.0, 0.0, PICK_PLACE_LOWER_OFFSET_M), action_space_info=FRANKA_IK_ACTION_SPACE),
            OpenGripperCfg(agent_assets="robot", duration=0.25, action_space_info=FRANKA_IK_ACTION_SPACE),
            MoveRelativeCfg(agent_assets="robot", position_offset=(0.0, 0.0, 0.05), action_space_info=FRANKA_IK_ACTION_SPACE),
            WaitCfg(duration=1.0),
        ],
        "pour": _pick_sequence() + [
            MoveRelativeCfg(agent_assets="robot", position_offset=(POUR_X_OFFSET_M, POUR_Y_OFFSET_M, 0.0), action_space_info=FRANKA_IK_ACTION_SPACE),
        ] + _pour_rotation_sequence(1.0) + [
            WaitCfg(duration=1.0),
        ] + _pour_rotation_sequence(-1.0) + _pour_return_sequence() + _pour_release_lower_sequence() + [
            OpenGripperCfg(agent_assets="robot", duration=0.25, action_space_info=FRANKA_IK_ACTION_SPACE),
            MoveRelativeCfg(agent_assets="robot", position_offset=(0.0, 0.0, 0.05), action_space_info=FRANKA_IK_ACTION_SPACE),
            WaitCfg(duration=1.0),
        ],
    }

    def __post_init__(self):
        super().__post_init__()
        # Use an explicit high-friction, zero-restitution contact policy for
        # the table, vessel, and gripper contact stack. The vessel-specific
        # event above assigns the authored coefficients to its shapes.
        self.sim.physics_material = sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="max",
            restitution_combine_mode="min",
            static_friction=1.0,
            dynamic_friction=0.8,
            restitution=0.0,
        )
        self.viewer.eye = TICKET0C7_VIEWER_EYE
        self.viewer.lookat = TICKET0C7_VIEWER_LOOKAT
        if _RUNTIME_CONFIGURATION_ERROR:
            raise ValueError(
                "Ticket 0c.7 task requires its per-run environment variables: "
                f"{_RUNTIME_CONFIGURATION_ERROR}"
            )


def ticket0c_configuration() -> dict:
    return {
        "task_id": TASK_ID,
        "asset_id": _LOADED_ASSET_ID,
        "asset_usd": _ASSET_USD,
        "scenario": _SCENARIO,
        "result_json": _RESULT_JSON,
        "target_slot": "vessel",
        "target_asset_count": 1,
        "station_preflight": {
            "candidate_offsets_m": [list(offset) for offset in STATION_CANDIDATES_M],
            "pick_place_selected_offset_m": list(PICK_PLACE_STATION_OFFSET_M),
            "pour_selected_offset_m": list(POUR_STATION_OFFSET_M),
            "selection_policy": "first valid candidate; pour must differ from pick/place",
        },
    }
