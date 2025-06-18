# Copyright (c) 2022-2025, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Franka Emika robots.

The following configurations are available:

* :obj:`FRANKA_ROBOTI2F85_INST_CFG`: Instantiated Franka Emika Panda robot with ROBOTIQ 2F85 gripper
* :obj:`FRANKA_ROBOTIQ2F85_INST_HIGH_PD_CFG`: Franka Emika Panda robot with Robotiq 2F85 gripper with stiffer PD control

Reference: https://github.com/frankaemika/franka_ros
"""
from matterix_assets import MATTERIX_ASSETS_DATA_DIR
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils import configclass
from ..matterix_articulation import MatterixArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.envs import mdp
from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.sensors import FrameTransformerCfg, OffsetCfg
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
##
# Configuration
##

marker_cfg = FRAME_MARKER_CFG.copy()
marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
marker_cfg.prim_path = "/Visuals/FrameTransformer"

@configclass
class FRANKA_ROBOTI2F85_INST_CFG(MatterixArticulationCfg):
    spawn = sim_utils.UsdFileCfg(
        usd_path=f"{MATTERIX_ASSETS_DATA_DIR}/robots/franka/franka-robotiq85/franka-robotiq85-inst.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    )
    init_state = ArticulationCfg.InitialStateCfg(
        joint_pos={
            "panda_joint1": 0.0,
            "panda_joint2": -0.569,
            "panda_joint3": 0.0,
            "panda_joint4": -2.810,
            "panda_joint5": 0.0,
            "panda_joint6": 3.037,
            "panda_joint7": 0.741,
            "finger_joint": 0.0,
        },
    )
    actuators = {
        "panda_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"],
            effort_limit_sim=87.0,
            velocity_limit_sim=2.175,
            stiffness=80.0,
            damping=4.0,
        ),
        "panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[5-7]"],
            effort_limit_sim=12.0,
            velocity_limit_sim=2.61,
            stiffness=80.0,
            damping=4.0,
        ),
        "robotiq_gripper": ImplicitActuatorCfg(
            joint_names_expr=["finger_joint"],
            effort_limit_sim=200.0,
            velocity_limit_sim=0.2,
            stiffness=2e3,
            damping=1e2,
        ),
    }
    soft_joint_pos_limit_factor = 1.0

    action_terms = {
        "arm_action": mdp.JointPositionActionCfg(
            joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
        ),
        "gripper_action": mdp.JointPositionActionCfg(
            joint_names=["finger_joint"],
        )
    }
    semantic_tags = [("class", "robot")]



"""Configuration of Franka Emika Panda robot with Robotiq 2F85 gripper."""


@configclass
class FRANKA_ROBOTIQ2F85_INST_HIGH_PD_CFG(FRANKA_ROBOTI2F85_INST_CFG):
    # Override `spawn` by copying and modifying the nested rigid_props
    spawn = FRANKA_ROBOTI2F85_INST_CFG().spawn.copy()
    spawn.rigid_props.disable_gravity = True

    # Copy and modify actuators
    actuators = FRANKA_ROBOTI2F85_INST_CFG().actuators.copy()
    actuators["panda_shoulder"] = actuators["panda_shoulder"].copy()
    actuators["panda_shoulder"].stiffness = 400.0
    actuators["panda_shoulder"].damping = 80.0

    actuators["panda_forearm"] = actuators["panda_forearm"].copy()
    actuators["panda_forearm"].stiffness = 400.0
    actuators["panda_forearm"].damping = 80.0


"""Configuration of Franka Emika Panda robot with  with Robotiq 2F85 gripper stiffer PD control.

This configuration is useful for task-space control using differential IK.
"""
@configclass
class FRANKA_PANDA_CFG(MatterixArticulationCfg):
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
    )
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
    )
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
    }
    soft_joint_pos_limit_factor=1.0
    """Configuration of Franka Emika Panda robot."""
    action_terms = {
            "arm_action": mdp.JointPositionActionCfg(
                joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
            ),
            "gripper_action": mdp.BinaryJointPositionActionCfg(
                joint_names=["panda_finger.*"],
                open_command_expr={"panda_finger_.*": 0.04},
                close_command_expr={"panda_finger_.*": 0.0},
            )
        }
    event_terms = {    
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

    sensors = {"ee_frame" : FrameTransformerCfg(
            prim_path="/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.1034],
                    ),
                ),
            ],
        )
    }
    semantic_tags = [("class", "robot")]

@configclass
class FRANKA_PANDA_HIGH_PD_CFG(FRANKA_PANDA_CFG):
    spawn = FRANKA_PANDA_CFG().spawn.copy()
    actuators = FRANKA_PANDA_CFG().actuators.copy()

    spawn.rigid_props.disable_gravity = True
    actuators["panda_shoulder"].stiffness = 400.0
    actuators["panda_shoulder"].damping = 80.0
    actuators["panda_forearm"].stiffness = 400.0
    actuators["panda_forearm"].damping = 80.0
    action_terms = {
            "arm_action": DifferentialInverseKinematicsActionCfg(
                asset_name="robot",
                joint_names=["panda_joint.*"],
                body_name="panda_hand",
                controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
                body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
            ),
            "gripper_action": mdp.BinaryJointPositionActionCfg(
                    joint_names=["panda_finger.*"],
                    open_command_expr={"panda_finger_.*": 0.04},
                    close_command_expr={"panda_finger_.*": 0.0},
                )
            }