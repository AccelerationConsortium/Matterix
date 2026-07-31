"""Dedicated physical checkpoint for the Corning 4980-50 flask."""

from matterix.envs import MatterixBaseEnvCfg, mdp
from matterix_assets.infrastructure.tables import TABLE_SEATTLE_INST_Cfg
from matterix_assets.labware.rigid_labware_batch1_local_only import CORNING_4980_50_LOCAL_ONLY_CFG
from matterix_assets.robots import FRANKA_PANDA_HIGH_PD_IK_CFG

from matterix_sm import PickObjectCfg
from matterix_sm.robot_action_spaces import FRANKA_IK_ACTION_SPACE

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

from .test_franka_rigid_labware_flasks import (
    FLASK_50_GRASP_OFFSET_M,
    FLASK_50_POS,
    EventCfg,
    _put_back,
)


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
        flask_50__object_world_pos = ObsTerm(
            func=mdp.object_world_pos, params={"asset_name": "flask_50"})
        flask_50__object_world_quat = ObsTerm(
            func=mdp.object_world_quat, params={"asset_name": "flask_50"})
        flask_50__object_lin_vel = ObsTerm(
            func=mdp.object_lin_vel, params={"asset_name": "flask_50"})
        flask_50__object_ang_vel = ObsTerm(
            func=mdp.object_ang_vel, params={"asset_name": "flask_50"})
        flask_50__pre_grasp_frame = ObsTerm(
            func=mdp.frame_world_pose,
            params={"asset_name": "flask_50", "frame_name": "pre_grasp"})
        flask_50__grasp_frame = ObsTerm(
            func=mdp.frame_world_pose,
            params={"asset_name": "flask_50", "frame_name": "grasp"})
        flask_50__post_grasp_frame = ObsTerm(
            func=mdp.frame_world_pose,
            params={"asset_name": "flask_50", "frame_name": "post_grasp"})
        flask_50__opening_frame = ObsTerm(
            func=mdp.frame_world_pose,
            params={"asset_name": "flask_50", "frame_name": "opening"})
        flask_50__base_frame = ObsTerm(
            func=mdp.frame_world_pose,
            params={"asset_name": "flask_50", "frame_name": "base"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    articulations: ArticulationsGroup = ArticulationsGroup()
    rigid_objects: RigidObjectsGroup = RigidObjectsGroup()


@configclass
class FrankaRigidLabwareFlask50EnvTestCfg(MatterixBaseEnvCfg):
    """One-environment-per-asset flask-50 visual checkpoint."""

    env_spacing = 10.0
    objects = {
        "flask_50": CORNING_4980_50_LOCAL_ONLY_CFG(pos=FLASK_50_POS),
        "table": TABLE_SEATTLE_INST_Cfg(pos=(0.5, 0, 0)),
    }
    articulated_assets = {
        "robot": FRANKA_PANDA_HIGH_PD_IK_CFG(pos=(0.0, 0, 0)),
    }
    gripper_joint_names = ["panda_finger_joint1", "panda_finger_joint2"]
    observations = ObservationManagerCfg()
    events = EventCfg()
    record_path = "datasets/dataset.hdf5"
    workflows = {
        "pickup_flask_50": PickObjectCfg(
            description="Pick up the 50 mL Erlenmeyer flask",
            agent_assets="robot",
            object="flask_50",
            action_space_info=FRANKA_IK_ACTION_SPACE,
        ),
        "pick_and_place_flask_50": [
            PickObjectCfg(
                description="Pick up the 50 mL Erlenmeyer flask",
                agent_assets="robot",
                object="flask_50",
                action_space_info=FRANKA_IK_ACTION_SPACE,
            ),
            *_put_back(FLASK_50_POS, FLASK_50_GRASP_OFFSET_M),
        ],
    }
