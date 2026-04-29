# Copyright (c) 2022-2026, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test development environment with multiple Franka robots and beakers."""

from matterix.envs import MatterixBaseEnvCfg, mdp
from matterix.managers.semantics.primitive_semantics import IsInContactPhysicsCfg
from matterix.managers.semantics.primitive_semantics.heat_transfer import (
    AmbientAirHeatConvectionCfg,
)
from matterix.managers.semantics.semantic_presets import HeaterCfg, HeatTransferCfg
from matterix_assets.equipment.ika_plate import IKA_PLATE_INST_CFG
from matterix_assets.infrastructure.tables import TABLE_SEATTLE_INST_Cfg
from matterix_assets.labware.beakers import BEAKER_500ML_INST_CFG
from matterix_assets.robots import FRANKA_PANDA_HIGH_PD_IK_CFG
from matterix_sm import PickObjectCfg, PlaceObjectCfg, TurnOnHeaterCfg, WaitCfg
from matterix_sm.robot_action_spaces import FRANKA_IK_ACTION_SPACE

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass


##
# Observation configs
##
@configclass
class ObservationManagerCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        ee_pos_robot = ObsTerm(func=mdp.ee_env_pos, params={"asset_name": "robot"})
        beaker_temperature = ObsTerm(func=mdp.object_temperature, params={"asset_name": "beaker"})
        table_temperature = ObsTerm(func=mdp.object_temperature, params={"asset_name": "table"})
        robot_temperature = ObsTerm(func=mdp.object_temperature, params={"asset_name": "robot"})
        beaker_is_in_contact = ObsTerm(func=mdp.object_is_in_contact, params={"asset_name": "beaker"})
        ika_plate_temperature = ObsTerm(func=mdp.object_temperature, params={"asset_name": "ika_plate"})
        ika_plate_is_heater_on = ObsTerm(func=mdp.object_is_heater_on, params={"asset_name": "ika_plate"})

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
            func=mdp.frame_world_pose,
            params={"asset_name": "beaker", "frame_name": "pre_grasp"},
        )
        beaker__grasp_frame = ObsTerm(
            func=mdp.frame_world_pose,
            params={"asset_name": "beaker", "frame_name": "grasp"},
        )
        beaker__post_grasp_frame = ObsTerm(
            func=mdp.frame_world_pose,
            params={"asset_name": "beaker", "frame_name": "post_grasp"},
        )

        # IKA plate observations
        ika_plate__object_world_pos = ObsTerm(func=mdp.object_world_pos, params={"asset_name": "ika_plate"})
        ika_plate__object_world_quat = ObsTerm(func=mdp.object_world_quat, params={"asset_name": "ika_plate"})

        # IKA plate placement frames
        ika_plate__pre_place_frame = ObsTerm(
            func=mdp.frame_world_pose,
            params={"asset_name": "ika_plate", "frame_name": "pre_place"},
        )
        ika_plate__place_frame = ObsTerm(
            func=mdp.frame_world_pose,
            params={"asset_name": "ika_plate", "frame_name": "place"},
        )
        ika_plate__post_place_frame = ObsTerm(
            func=mdp.frame_world_pose,
            params={"asset_name": "ika_plate", "frame_name": "post_place"},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

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

    policy: PolicyCfg = PolicyCfg()
    rigid_objects: RigidObjectsGroup = RigidObjectsGroup()
    articulations: ArticulationsGroup = ArticulationsGroup()


##
# Test Development Environment configs
# Environment with multiple agents (robots), multiple beakers, and tables.
##
@configclass
class FrankaBeakerHeaterSemanticsEnvTestCfg(MatterixBaseEnvCfg):
    env_spacing = 5.0
    episode_length_s = 100.0  # sec
    gripper_joint_names = ["panda_finger_joint1", "panda_finger_joint2"]

    objects = {
        "beaker": BEAKER_500ML_INST_CFG(
            pos=(0.6, 0.05, 0.05),
            semantics=[
                # Relative prim paths: "asset_name[/body_link]"
                # setup_scene() prepends {ENV_REGEX_NS}/Articulations_ or /RigidObjects_ automatically.
                IsInContactPhysicsCfg(
                    filter_prim_paths_expr=[
                        "ika_plate",
                    ],
                    verbose=True,
                ),
                HeatTransferCfg(temp=298.15, C=7920.0, A=0.1, K=200, verbose=True),
            ],
        ),
        "ika_plate": IKA_PLATE_INST_CFG(
            pos=(0.4, -0.3, 0.12),
            rot=(0, 0, 0, 1),
            semantics=HeaterCfg(
                temp=350.15,
                C=7920.0,
                A=0.5,
                heater_on=False,
                target_temperature=298.15,
                K_heater=0.1,
                verbose=True,
            ),
        ),
        "table": TABLE_SEATTLE_INST_Cfg(
            pos=(0.5, 0, 0),
            mass=50.0,
            semantics=HeatTransferCfg(temp=323.15, C=7920.0, A=1.1, verbose=True),
        ),
    }

    articulated_assets = {
        "robot": FRANKA_PANDA_HIGH_PD_IK_CFG(
            pos=(0.0, 0, 0),
            mass=20.0,
            semantics=HeatTransferCfg(temp=350.15, C=7920.0, A=0.5, verbose=True),
        ),
    }

    observations = ObservationManagerCfg()

    semantics = [AmbientAirHeatConvectionCfg(verbose=True)]

    # Define available workflows for this environment
    workflows = {
        "pickup_beaker": [
            PickObjectCfg(
                description="Pick up the beaker and observe heat transfer while holding",
                agent_assets="robot",
                object="beaker",
                action_space_info=FRANKA_IK_ACTION_SPACE,
            ),
            WaitCfg(duration=5.0),  # Hold the beaker and observe temperature change
        ],
        "pickup_and_place_beaker": [
            WaitCfg(duration=2.0),  # Wait and observe heat transfer from IKA plate to beaker,
            TurnOnHeaterCfg(
                asset_name="ika_plate",
                value=True,
                target_temperature=373.15,
            ),
            # Add heater state semantics to observe target temperature changes
            PickObjectCfg(
                description="Pick up the beaker",
                agent_assets="robot",
                object="beaker",
                action_space_info=FRANKA_IK_ACTION_SPACE,
            ),
            PlaceObjectCfg(
                description="Place the beaker on top of the IKA plate",
                agent_assets="robot",
                target="ika_plate",
                action_space_info=FRANKA_IK_ACTION_SPACE,
            ),
            WaitCfg(duration=10.0),  # Wait and observe heat transfer from IKA plate to beaker
            TurnOnHeaterCfg(
                asset_name="ika_plate",
                value=False,
            ),
            WaitCfg(duration=5.0),  # Wait and observe heat transfer from IKA plate to beaker,
        ],
    }

    def __post_init__(self):
        super().__post_init__()
        self.events.randomize_table_temp = EventTerm(
            func=mdp.randomize_temperature,
            mode="reset",
            params={
                "asset_name": "beaker",
                "min_temp": 293.15,
                "max_temp": 323.15,
            },  # 20-50°C
        )
        self.events.randomize_robot_temp = EventTerm(
            func=mdp.randomize_temperature,
            mode="reset",
            params={
                "asset_name": "robot",
                "min_temp": 340.15,
                "max_temp": 360.15,
            },  # 67-87°C
        )
        self.events.randomize_beaker_pos = EventTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg("beaker"),
            },
        )
