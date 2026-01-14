# Copyright (c) 2022-2026, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test development environment with Franka robot, beaker, and table for lifting tasks."""

from matterix.envs import MatterixBaseEnvCfg, mdp
from matterix.managers import EventManagerCfg
from matterix_assets.infrastructure.tables import TABLE_SEATTLE_INST_Cfg
from matterix_assets.labware.beakers import BEAKER_500ML_INST_CFG
from matterix_assets.robots import FRANKA_PANDA_HIGH_PD_IK_CFG

# Import workflow definitions from separate file (avoids circular deps and allows lightweight listing)
from matterix_sm import PickObjectCfg
from matterix_sm.robot_action_spaces import FRANKA_IK_ACTION_SPACE

import isaaclab.envs.mdp as isaaclab_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass


##
# Event configs
##
@configclass
class EventCfg(EventManagerCfg):
    """Configuration for randomization events."""

    # Reset scene to default state
    reset_scene_to_default = EventTerm(
        func=isaaclab_mdp.reset_scene_to_default,
        mode="reset",
    )

    # Randomize beaker position on reset
    randomize_beaker_position = EventTerm(
        func=isaaclab_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.1, 0.1),  # Offset from default position (0.6)
                "y": (-0.15, 0.15),  # Offset from default position (0.05)
                "z": (0.0, 0.0),  # Keep default Z height
            },
            "velocity_range": {},  # No initial velocity
            "asset_cfg": SceneEntityCfg("beaker"),
        },
    )


##
# Observation configs
##
@configclass
class ObservationManagerCfg:
    """Observation specifications for the MDP."""

    @configclass
    class ArticulationsGroup(ObsGroup):
        """Articulations group with robot observations using dotted keys."""

        # Robot observations - keys create nested structure: obs["articulations"]["robot"][key]
        robot__root_world_pos = ObsTerm(func=mdp.root_world_pos, params={"asset_name": "robot"})
        robot__root_world_quat = ObsTerm(func=mdp.root_world_quat, params={"asset_name": "robot"})
        robot__joint_pos = ObsTerm(func=mdp.joint_pos, params={"asset_name": "robot"})
        robot__joint_vel = ObsTerm(func=mdp.joint_vel, params={"asset_name": "robot"})
        robot__ee_world_pos = ObsTerm(func=mdp.ee_world_pos, params={"asset_name": "robot"})
        robot__ee_world_quat = ObsTerm(func=mdp.ee_world_quat, params={"asset_name": "robot"})
        robot__gripper_pos = ObsTerm(func=mdp.gripper_pos, params={"asset_name": "robot"})
        robot__grasping_frame_world_pos = ObsTerm(
            func=mdp.frame_world_pos,
            params={"asset_name": "robot", "frame_name": "grasping_frame"},
        )
        robot__grasping_frame_world_quat = ObsTerm(
            func=mdp.frame_world_quat,
            params={"asset_name": "robot", "frame_name": "grasping_frame"},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class RigidObjectsGroup(ObsGroup):
        """Rigid objects group with beaker observations using dotted keys."""

        # Beaker observations - keys create nested structure: obs["rigid_objects"]["beaker"][key]
        beaker__object_world_pos = ObsTerm(func=mdp.object_world_pos, params={"asset_name": "beaker"})
        beaker__object_world_quat = ObsTerm(func=mdp.object_world_quat, params={"asset_name": "beaker"})
        beaker__object_lin_vel = ObsTerm(func=mdp.object_lin_vel, params={"asset_name": "beaker"})
        beaker__object_ang_vel = ObsTerm(func=mdp.object_ang_vel, params={"asset_name": "beaker"})

        # Frame transformations (continuously computed in world frame as 7D poses)
        beaker__pre_grasp_frame = ObsTerm(
            func=mdp.frame_world_pose, params={"asset_name": "beaker", "frame_name": "pre_grasp"}
        )
        beaker__grasp_frame = ObsTerm(func=mdp.frame_world_pose, params={"asset_name": "beaker", "frame_name": "grasp"})
        beaker__post_grasp_frame = ObsTerm(
            func=mdp.frame_world_pose, params={"asset_name": "beaker", "frame_name": "post_grasp"}
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    articulations: ArticulationsGroup = ArticulationsGroup()
    rigid_objects: RigidObjectsGroup = RigidObjectsGroup()


##
# Test Development Environment configs
# Environment with Franka robot, beaker, and table for lifting tasks.
##
@configclass
class FrankaBeakerLiftEnvTestCfg(MatterixBaseEnvCfg):
    env_spacing = 5.0

    objects = {
        "beaker": BEAKER_500ML_INST_CFG(pos=(0.6, 0.05, 0.05)),
        "table": TABLE_SEATTLE_INST_Cfg(pos=(0.5, 0, 0)),
    }

    articulated_assets = {
        "robot": FRANKA_PANDA_HIGH_PD_IK_CFG(pos=(0.0, 0, 0)),  # robot arm with joint controller
    }

    # Gripper joint names for observation functions
    gripper_joint_names = ["panda_finger_joint1", "panda_finger_joint2"]

    observations = ObservationManagerCfg()
    events = EventCfg()

    record_path = "datasets/dataset.hdf5"

    # Define available workflows for this environment
    workflows = {
        "pickup_beaker": PickObjectCfg(
            description="Pick up the beaker",
            agent_assets="robot",
            object="beaker",
            action_space_info=FRANKA_IK_ACTION_SPACE,
        )
    }
