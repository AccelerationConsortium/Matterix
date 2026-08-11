"""Dedicated physical checkpoint for the Falcon 15 mL capped tube."""

from matterix.envs import MatterixBaseEnvCfg
from isaaclab.utils import configclass

from .capped_labware_checkpoints import capped_env_fields


@configclass
class FrankaRigidLabwareFalcon15EnvTestCfg(MatterixBaseEnvCfg):
    """One-environment-per-asset Falcon 15 mL visual checkpoint."""

    locals().update(capped_env_fields(
        slug="falcon-352096",
        vessel_pos=(0.55, 0.0, 0.0),
        pre_grasp_z=0.1438,
        grasp_z=0.104,
        workflow_name="pick_and_place_falcon_15",
        description="Pick up and place the fixed-jointed Falcon 15 mL tube",
    ))
