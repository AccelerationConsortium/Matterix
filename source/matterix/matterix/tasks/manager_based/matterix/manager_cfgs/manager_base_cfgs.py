
from isaaclab.utils import configclass


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""


@configclass
class EventCfg:
    """Configuration for events."""


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    # @configclass
    # class PolicyCfg(ObsGroup):
    #     """Observations for policy group with state values."""

    #     actions = ObsTerm(func=mdp.last_action)
    #     joint_pos = ObsTerm(func=mdp.joint_pos_rel)
    #     joint_vel = ObsTerm(func=mdp.joint_vel_rel)
    #     object = ObsTerm(func=mdp.object_obs)
    #     cube_positions = ObsTerm(func=mdp.cube_positions_in_world_frame)
    #     cube_orientations = ObsTerm(func=mdp.cube_orientations_in_world_frame)
    #     eef_pos = ObsTerm(func=mdp.ee_frame_pos)
    #     eef_quat = ObsTerm(func=mdp.ee_frame_quat)
    #     gripper_pos = ObsTerm(func=mdp.gripper_pos)

    #     def __post_init__(self):
    #         self.enable_corruption = False
    #         self.concatenate_terms = False
