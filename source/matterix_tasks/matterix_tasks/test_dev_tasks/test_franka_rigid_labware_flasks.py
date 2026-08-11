# Copyright (c) 2022-2026, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared helpers for the per-asset flask WebRTC checkpoint environments.

Each registered task imports these frame observations and placement helpers while
keeping exactly one flask in its own environment.
"""

from matterix.envs import MatterixBaseEnvCfg, mdp
from matterix.managers import EventManagerCfg
from matterix_assets.infrastructure.tables import TABLE_SEATTLE_INST_Cfg
from matterix_assets.labware.rigid_labware_batch1_local_only import (
    CORNING_4980_50_LOCAL_ONLY_CFG,
    CORNING_4980_250_LOCAL_ONLY_CFG,
)
from matterix_assets.robots import FRANKA_PANDA_HIGH_PD_IK_CFG

import torch

from matterix_sm import MoveRelativeCfg, OpenGripperCfg, PickObjectCfg
from matterix_sm.primitive_actions.move_to_pose import MoveToPoseCfg
from matterix_sm.robot_action_spaces import FRANKA_IK_ACTION_SPACE

import isaaclab.envs.mdp as isaaclab_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass


@configclass
class EventCfg(EventManagerCfg):
    """Reset events for the dedicated flask checkpoint."""

    reset_scene_to_default = EventTerm(
        func=isaaclab_mdp.reset_scene_to_default,
        mode="reset",
    )


APPROACH_CLEARANCE_M = 0.15
RELEASE_CLEARANCE_M = 0.005
FLASK_50_POS = (0.55, 0.0, 0.039)
FLASK_250_POS = (0.55, 0.24, 0.066)
FLASK_50_GRASP_OFFSET_M = 0.031
FLASK_250_GRASP_OFFSET_M = 0.050


def _put_back(vessel_pos, grasp_offset_z, agent="robot"):
    """Return a flask to its authored world pose and release it."""
    x, y, z = vessel_pos
    return [
        MoveToPoseCfg(
            agent_assets=agent,
            target_positions=torch.tensor([[x, y, z + grasp_offset_z + APPROACH_CLEARANCE_M]]),
            action_space_info=FRANKA_IK_ACTION_SPACE,
        ),
        MoveToPoseCfg(
            agent_assets=agent,
            target_positions=torch.tensor([[x, y, z + grasp_offset_z + RELEASE_CLEARANCE_M]]),
            action_space_info=FRANKA_IK_ACTION_SPACE,
        ),
        OpenGripperCfg(agent_assets=agent, action_space_info=FRANKA_IK_ACTION_SPACE),
        MoveRelativeCfg(
            agent_assets=agent,
            position_offset=(0.0, 0.0, APPROACH_CLEARANCE_M),
            orientation_offset=None,
            action_space_info=FRANKA_IK_ACTION_SPACE,
        ),
    ]


@configclass
class ObservationManagerCfg:
    """Robot and flask-frame observations consumed by the state machine."""

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

        flask_250__object_world_pos = ObsTerm(
            func=mdp.object_world_pos, params={"asset_name": "flask_250"})
        flask_250__object_world_quat = ObsTerm(
            func=mdp.object_world_quat, params={"asset_name": "flask_250"})
        flask_250__object_lin_vel = ObsTerm(
            func=mdp.object_lin_vel, params={"asset_name": "flask_250"})
        flask_250__object_ang_vel = ObsTerm(
            func=mdp.object_ang_vel, params={"asset_name": "flask_250"})
        flask_250__pre_grasp_frame = ObsTerm(
            func=mdp.frame_world_pose,
            params={"asset_name": "flask_250", "frame_name": "pre_grasp"})
        flask_250__grasp_frame = ObsTerm(
            func=mdp.frame_world_pose,
            params={"asset_name": "flask_250", "frame_name": "grasp"})
        flask_250__post_grasp_frame = ObsTerm(
            func=mdp.frame_world_pose,
            params={"asset_name": "flask_250", "frame_name": "post_grasp"})
        flask_250__opening_frame = ObsTerm(
            func=mdp.frame_world_pose,
            params={"asset_name": "flask_250", "frame_name": "opening"})
        flask_250__base_frame = ObsTerm(
            func=mdp.frame_world_pose, params={"asset_name": "flask_250", "frame_name": "base"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    articulations: ArticulationsGroup = ArticulationsGroup()
    rigid_objects: RigidObjectsGroup = RigidObjectsGroup()
