"""Shared helpers for capped DURAN and Falcon visual checkpoints.

The capped payloads are intentionally spawned as generic USD assets.  Each USD
contains two rigid bodies joined by a ``PhysicsFixedJoint`` and does not expose
an articulation root, so it must not be registered as a ``RigidObject``.
"""

import torch

from matterix.envs import MatterixBaseEnvCfg, mdp
from matterix_assets import MATTERIX_ASSETS_DATA_DIR, MatterixStaticObjectCfg
from matterix_assets.infrastructure.tables import TABLE_SEATTLE_INST_Cfg
from matterix_assets.robots import FRANKA_PANDA_HIGH_PD_IK_CFG

from matterix.managers import EventManagerCfg
from matterix_sm import CloseGripperCfg, MoveRelativeCfg, OpenGripperCfg
from matterix_sm.primitive_actions.move_to_pose import MoveToPoseCfg
from matterix_sm.robot_action_spaces import FRANKA_IK_ACTION_SPACE

import isaaclab.envs.mdp as isaaclab_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass


@configclass
class EventCfg(EventManagerCfg):
    """Reset events for a capped labware checkpoint."""

    reset_scene_to_default = EventTerm(
        func=isaaclab_mdp.reset_scene_to_default,
        mode="reset",
    )


@configclass
class ObservationManagerCfg:
    """Robot observations used by the hard-coded visual manipulation sequence."""

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

    articulations: ArticulationsGroup = ArticulationsGroup()


def capped_payload_cfg(slug: str, pos: tuple[float, float, float]) -> MatterixStaticObjectCfg:
    """Build a generic scene asset for one two-body capped USD payload."""
    return MatterixStaticObjectCfg(
        usd_path=f"{MATTERIX_ASSETS_DATA_DIR}/labware/{slug}/{slug}-inst.usda",
        pos=pos,
        scale=(1.0, 1.0, 1.0),
    )


def capped_pick_and_place(
    vessel_pos: tuple[float, float, float],
    pre_grasp_z: float,
    grasp_z: float,
    agent: str = "robot",
):
    """Return the visual checkpoint sequence for one fixed-jointed payload."""
    x, y, z = vessel_pos
    return [
        OpenGripperCfg(agent_assets=agent, action_space_info=FRANKA_IK_ACTION_SPACE),
        MoveToPoseCfg(
            agent_assets=agent,
            target_positions=torch.tensor([[x, y, z + pre_grasp_z]]),
            action_space_info=FRANKA_IK_ACTION_SPACE,
        ),
        MoveToPoseCfg(
            agent_assets=agent,
            target_positions=torch.tensor([[x, y, z + grasp_z]]),
            action_space_info=FRANKA_IK_ACTION_SPACE,
        ),
        CloseGripperCfg(agent_assets=agent, action_space_info=FRANKA_IK_ACTION_SPACE),
        MoveRelativeCfg(
            agent_assets=agent,
            position_offset=(0.0, 0.0, 0.1),
            orientation_offset=None,
            action_space_info=FRANKA_IK_ACTION_SPACE,
        ),
        MoveToPoseCfg(
            agent_assets=agent,
            target_positions=torch.tensor([[x, y, z + grasp_z + 0.005]]),
            action_space_info=FRANKA_IK_ACTION_SPACE,
        ),
        OpenGripperCfg(agent_assets=agent, action_space_info=FRANKA_IK_ACTION_SPACE),
        MoveRelativeCfg(
            agent_assets=agent,
            position_offset=(0.0, 0.0, 0.15),
            orientation_offset=None,
            action_space_info=FRANKA_IK_ACTION_SPACE,
        ),
    ]


def capped_env_fields(
    slug: str,
    vessel_pos: tuple[float, float, float],
    pre_grasp_z: float,
    grasp_z: float,
    workflow_name: str,
    description: str,
) -> dict:
    """Return common config fields for a one-asset capped checkpoint."""
    return {
        "env_spacing": 10.0,
        "objects": {
            "capped_labware": capped_payload_cfg(slug, vessel_pos),
            "table": TABLE_SEATTLE_INST_Cfg(pos=(0.5, 0, 0)),
        },
        "articulated_assets": {
            "robot": FRANKA_PANDA_HIGH_PD_IK_CFG(pos=(0.0, 0, 0)),
        },
        "gripper_joint_names": ["panda_finger_joint1", "panda_finger_joint2"],
        "observations": ObservationManagerCfg(),
        "events": EventCfg(),
        "record_path": "datasets/dataset.hdf5",
        "workflows": {
            workflow_name: {
                "description": description,
                "actions": capped_pick_and_place(vessel_pos, pre_grasp_z, grasp_z),
            }
        },
    }
