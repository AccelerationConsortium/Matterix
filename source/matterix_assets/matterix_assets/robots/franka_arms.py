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

##
# Configuration
##


@configclass
class FRANKA_ROBOTI2F85_INST_CFG(ArticulationCfg):
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


"""Configuration of Franka Emika Panda robot with Robotiq 2F85 gripper."""


@configclass
class FRANKA_ROBOTIQ2F85_INST_HIGH_PD_CFG(FRANKA_ROBOTI2F85_INST_CFG):
    # Override `spawn` by copying and modifying the nested rigid_props
    spawn = FRANKA_ROBOTI2F85_INST_CFG.spawn.copy()
    spawn.rigid_props.disable_gravity = True

    # Copy and modify actuators
    actuators = FRANKA_ROBOTI2F85_INST_CFG.actuators.copy()
    actuators["panda_shoulder"] = actuators["panda_shoulder"].copy()
    actuators["panda_shoulder"].stiffness = 400.0
    actuators["panda_shoulder"].damping = 80.0

    actuators["panda_forearm"] = actuators["panda_forearm"].copy()
    actuators["panda_forearm"].stiffness = 400.0
    actuators["panda_forearm"].damping = 80.0


"""Configuration of Franka Emika Panda robot with  with Robotiq 2F85 gripper stiffer PD control.

This configuration is useful for task-space control using differential IK.
"""
