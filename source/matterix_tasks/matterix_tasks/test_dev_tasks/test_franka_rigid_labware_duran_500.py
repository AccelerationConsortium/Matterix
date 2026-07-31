"""Dedicated physical checkpoint for the DURAN 500 mL capped bottle."""

from matterix.envs import MatterixBaseEnvCfg
from isaaclab.utils import configclass

from .capped_labware_checkpoints import capped_env_fields


@configclass
class FrankaRigidLabwareDuran500EnvTestCfg(MatterixBaseEnvCfg):
    """One-environment-per-asset DURAN 500 mL visual checkpoint."""

    locals().update(capped_env_fields(
        slug="dwk-218014459",
        vessel_pos=(0.55, 0.0, 0.0),
        pre_grasp_z=0.231,
        grasp_z=0.145,
        workflow_name="pick_and_place_duran_500",
        description="Pick up and place the fixed-jointed DURAN 500 mL GL45 bottle",
    ))
