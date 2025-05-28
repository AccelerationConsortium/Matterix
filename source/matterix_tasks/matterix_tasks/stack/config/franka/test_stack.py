# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import RigidObjectCfg, AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.manager_based.manipulation.stack import mdp
from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events
from isaaclab.envs import TestBaseEnvCfg
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

FRANKA_PANDA_CFG_85 = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/home/arjun/Desktop/Matterix_assets/robots/franka/franka-robotiq85/franka-robotiq85_inst.usda",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "panda_joint1": 0.0,
            "panda_joint2": -0.569,
            "panda_joint3": 0.0,
            "panda_joint4": -2.810,
            "panda_joint5": 0.0,
            "panda_joint6": 3.037,
            "panda_joint7": 0.741,
            "panda_joint8": 0.0,
            "dynamixel_robotiq_joint": 0.0,
            "dynamixel_robotiq_coupling": 0.0,
            "panda_dynamixel_coupling" : 0.0,
            "dynamixel_base_joint" : 0.0,
            "robotiq_85_base_joint" : 0.0,
            "robotiq_85_left_knuckle_joint" : 0.0,
            "robotiq_85_right_knuckle_joint" : 0.0,
            "robotiq_85_left_finger_joint" : 0.0,
            "robotiq_85_right_finger_joint" : 0.0,
            "robotiq_85_left_inner_knuckle_joint" : 0.0,
            "robotiq_85_right_inner_knuckle_joint" : 0.0,
            "robotiq_85_left_finger_tip_joint" : 0.0,
            "robotiq_85_right_finger_tip_joint" : 0.0,
        },
    ),
    actuators={
        "panda_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"],
            effort_limit=87.0,
            velocity_limit=100.0,
            stiffness=80.0,
            damping=4.0,
        ),
        "panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[5-7]"],
            effort_limit=12.0,
            velocity_limit=100.0,
            stiffness=80.0,
            damping=4.0,
        ),
        "robotiq_gripper": ImplicitActuatorCfg(
            joint_names_expr=["robotiq_85_left_knuckle_joint", "robotiq_85_right_knuckle_joint", "robotiq_85_left_finger_joint", "robotiq_85_right_finger_joint", "robotiq_85_left_inner_knuckle_joint", "robotiq_85_right_inner_knuckle_joint", "robotiq_85_left_finger_tip_joint", "robotiq_85_right_finger_tip_joint"],
            effort_limit=1000.0,
            velocity_limit=2.0,
            stiffness=2e3,
            damping=1e2,
        )
    },
    soft_joint_pos_limit_factor=1.0,
)
FRANKA_PANDA_CFG_85.action_terms = {
        "arm_action": mdp.JointPositionActionCfg(
            asset_name="robot2", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
        ),
        "gripper_action": mdp.BinaryJointPositionActionCfg(
            asset_name="robot2",
            joint_names=["robotiq_85_.*"],
            open_command_expr={"robotiq_85_.*": 0.04},
            close_command_expr={"robotiq_85_.*": 0.0},
        )
    }

FRANKA_PANDA_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "panda_joint1": 0.0,
            "panda_joint2": -0.569,
            "panda_joint3": 0.0,
            "panda_joint4": -2.810,
            "panda_joint5": 0.0,
            "panda_joint6": 3.037,
            "panda_joint7": 0.741,
            "panda_finger_joint.*": 0.04,
        },
    ),
    actuators={
        "panda_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"],
            effort_limit=87.0,
            velocity_limit=2.175,
            stiffness=80.0,
            damping=4.0,
        ),
        "panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[5-7]"],
            effort_limit=12.0,
            velocity_limit=2.61,
            stiffness=80.0,
            damping=4.0,
        ),
        "panda_hand": ImplicitActuatorCfg(
            joint_names_expr=["panda_finger_joint.*"],
            effort_limit=200.0,
            velocity_limit=0.2,
            stiffness=2e3,
            damping=1e2,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of Franka Emika Panda robot."""
FRANKA_PANDA_CFG.action_terms = {
        "arm_action": mdp.JointPositionActionCfg(
            asset_name="robot2", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
        ),
        "gripper_action": mdp.BinaryJointPositionActionCfg(
            asset_name="robot2",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )
    }
FRANKA_PANDA_CFG.event_terms = {    
    "init_franka_arm_pose" : EventTerm(
        func=franka_stack_events.set_default_joint_pose,
        mode="startup",
        params={
            "default_pose": [0.0444, -0.1894, -0.1107, -2.5148, 0.0044, 2.3775, 0.6952, 0.0400, 0.0400],
        },
    ),

    "randomize_franka_joint_state" : EventTerm(
        func=franka_stack_events.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={
            "mean": 0.0,
            "std": 0.02,
        },
    )
}

FRANKA_PANDA_CFG.semantic_tags = [("class", "robot")]
##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip

cube_properties = RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        )

@configclass
class FrankaCubeStackEnvTestCfg(TestBaseEnvCfg):

    objects = {
        # Set each stacking cube deterministically
        "cube_1" : RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube_1",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.4, 0.0, 0.0203], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/blue_block.usd",
                scale=(1.0, 1.0, 1.0),
                rigid_props=cube_properties,
                semantic_tags=[("class", "cube_1")],
            ),
        ),
        "cube_2" : RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube_2",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.55, 0.05, 0.0203], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/red_block.usd",
                scale=(1.0, 1.0, 1.0),
                rigid_props=cube_properties,
                semantic_tags=[("class", "cube_2")],
            ),
        ),
        "cube_3" : RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube_3",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.60, -0.1, 0.0203], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/green_block.usd",
                scale=(1.0, 1.0, 1.0),
                rigid_props=cube_properties,
                semantic_tags=[("class", "cube_3")],
            ),
        ),
        "table" : AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Table",
            init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
            spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
        )
    }

    articulated_assets = {
        "robot" : FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot"),
        "robot2" : FRANKA_PANDA_CFG_85.replace(prim_path="{ENV_REGEX_NS}/Robot2")
    }
    articulated_assets["robot2"].init_state.pos = (1, 0, 0)

