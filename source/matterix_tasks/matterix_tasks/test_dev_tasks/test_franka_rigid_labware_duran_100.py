"""Dedicated physical checkpoint for the DURAN 100 mL capped bottle."""

from matterix.envs import MatterixBaseEnvCfg
from isaaclab.utils import configclass

from .capped_labware_checkpoints import capped_env_fields


@configclass
class FrankaRigidLabwareDuran100EnvTestCfg(MatterixBaseEnvCfg):
    """One-environment-per-asset DURAN 100 mL visual checkpoint."""

    locals().update(capped_env_fields(
        slug="dwk-218012458",
        vessel_pos=(0.55, 0.0, 0.0),
        pre_grasp_z=0.145,
        grasp_z=0.075,
        workflow_name="pick_and_place_duran_100",
        description="Pick up and place the fixed-jointed DURAN 100 mL GL45 bottle",
    ))
