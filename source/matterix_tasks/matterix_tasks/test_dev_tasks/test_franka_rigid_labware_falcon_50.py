"""Dedicated physical checkpoint for the Falcon 50 mL capped tube."""

from matterix.envs import MatterixBaseEnvCfg
from isaaclab.utils import configclass

from .capped_labware_checkpoints import capped_env_fields


@configclass
class FrankaRigidLabwareFalcon50EnvTestCfg(MatterixBaseEnvCfg):
    """One-environment-per-asset Falcon 50 mL visual checkpoint."""

    locals().update(capped_env_fields(
        slug="falcon-352070",
        vessel_pos=(0.55, 0.0, 0.0),
        pre_grasp_z=0.13955,
        grasp_z=0.097,
        workflow_name="pick_and_place_falcon_50",
        description="Pick up and place the fixed-jointed Falcon 50 mL tube",
    ))
