# Copyright (c) 2022-2026, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base configuration of the environment.

This module defines the general configuration of the environment. It includes parameters for
configuring the environment instances, viewer settings, and simulation parameters.
"""

from dataclasses import MISSING
from typing import Literal

from isaaclab_physx.physics import PhysxCfg
from matterix.envs import mdp
from matterix.managers import (
    ActionManagerCfg,
    DefaultEventManagerCfg,
    MatterixBaseRecorderCfg,
    ObservationManagerCfg,
)
from matterix.managers.semantics import SemanticsCfg
from matterix.particle_systems import ParticleSystemCfg, ReservedParticleCfg
from matterix_assets import (
    MatterixArticulationCfg,
    MatterixRigidObjectCfg,
    MatterixStaticObjectCfg,
)

from isaaclab.envs.common import ViewerCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import SensorBaseCfg
from isaaclab.sim import RenderCfg, SimulationCfg
from isaaclab.sim.spawners.lights import DomeLightCfg, LightCfg
from isaaclab.utils import configclass


@configclass
class LightStateCfg:
    """Configuration of the environment lights."""

    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rot: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)  # (x, y, z, w)
    light: LightCfg = MISSING


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Terminate if the episode length is exceeded
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class MatterixBaseEnvCfg:
    """Base configuration of the environment."""

    # simulation settings
    viewer: ViewerCfg = ViewerCfg()
    """Viewer configuration. Default is ViewerCfg()."""

    sim: SimulationCfg = SimulationCfg(
        render=RenderCfg(
            # "rtx_raytracing_fractionalCutoutOpacity" removed -- cleanup of a dead
            # setting, not an active-crash fix on current main. On isaaclab==2.3.0
            # (Isaac Sim 5.1.0, what main was pinned to when this was added in #23),
            # SimulationContext._apply_render_settings_from_cfg() validated every
            # carb_settings key against currently-registered carb settings and raised
            # ValueError when a key wasn't found -- this key isn't registered under
            # Isaac Sim 5.1.0 and blocked all env construction. Confirmed the setting
            # has no home anywhere under isaacsim 6.0.1.0 either (full /rtx settings
            # tree search, including the renderer's newer rtpt/path-tracing branch),
            # so this isn't a rename to chase -- the feature appears retired.
            # On current main's isaaclab==3.0.0b2.post1, the equivalent code
            # (SimulationContext._apply_render_settings()) dropped that validation
            # entirely and just calls set_setting() unconditionally, so the same dead
            # key no longer raises -- it's a silent no-op there instead of a crash.
            # Left out regardless, since it does nothing on any version this project
            # has targeted. rtx_translucency_enabled is unaffected and still applies.
            carb_settings={"rtx_translucency_enabled": True}
        )
    )
    """Physics simulation configuration. Default is SimulationCfg()."""

    # ui settings
    ui_window_class_type: type | None = BaseEnvWindow
    """The class type of the UI window. Default is None.

    If None, then no UI window is created.

    Note:
        If you want to make your own UI window, you can create a class that inherits from
        from :class:`isaaclab.envs.ui.base_env_window.BaseEnvWindow`. Then, you can set
        this attribute to your class type.
    """

    # general settings
    seed: int | None = None
    """The seed for the random number generator. Defaults to None, in which case the seed is not set.

    Note:
      The seed is set at the beginning of the environment initialization. This ensures that the environment
      creation is deterministic and behaves similarly across different runs.
    """

    decimation: int = 5
    """Number of control action updates @ sim dt per policy dt.

    For instance, if the simulation dt is 0.01s and the policy dt is 0.1s, then the decimation is 10.
    This means that the control action is updated every 10 simulation steps.
    """

    # environment settings
    scene: InteractiveSceneCfg = MISSING
    """Scene settings.

    Please refer to the :class:`isaaclab.scene.InteractiveSceneCfg` class for more details.
    """

    # light
    lights: dict[str, LightStateCfg] = {
        "light": LightStateCfg(light=DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0))
    }

    recorders: MatterixBaseRecorderCfg = MatterixBaseRecorderCfg()
    """Recorder settings. Defaults to recording nothing.

    Please refer to the :class:`isaaclab.managers.RecorderManager` class for more details.
    """
    record_path: str = None
    # Matterix owns recording through RealTimeVideoRecorder. Disable Isaac Lab's
    # separate recorder while retaining the config field required by its base env.
    video_recorder: object | None = None

    observations: object = ObservationManagerCfg()
    """Observation space settings.

    Please refer to the :class:`isaaclab.managers.ObservationManager` class for more details.
    """

    actions: object = ActionManagerCfg()
    """Action space settings.

    Please refer to the :class:`isaaclab.managers.ActionManager` class for more details.
    """

    events: object = DefaultEventManagerCfg()
    """Event settings. Defaults to the basic configuration that resets the scene to its default state.

    Please refer to the :class:`isaaclab.managers.EventManager` class for more details.
    """

    # general settings
    is_finite_horizon: bool = False
    """Whether the learning task is treated as a finite or infinite horizon problem for the agent.
    Defaults to False, which means the task is treated as an infinite horizon problem.

    This flag handles the subtleties of finite and infinite horizon tasks:

    * **Finite horizon**: no penalty or bootstrapping value is required by the the agent for
      running out of time. However, the environment still needs to terminate the episode after the
      time limit is reached.
    * **Infinite horizon**: the agent needs to bootstrap the value of the state at the end of the episode.
      This is done by sending a time-limit (or truncated) done signal to the agent, which triggers this
      bootstrapping calculation.

    If True, then the environment is treated as a finite horizon problem and no time-out (or truncated) done signal
    is sent to the agent. If False, then the environment is treated as an infinite horizon problem and a time-out
    (or truncated) done signal is sent to the agent.

    Note:
        The base :class:`ManagerBasedRLEnv` class does not use this flag directly. It is used by the environment
        wrappers to determine what type of done signal to send to the corresponding learning agent.
    """

    episode_length_s: float = 30.0
    """Duration of an episode (in seconds).

    Based on the decimation rate and physics time step, the episode length is calculated as:

    .. code-block:: python

        episode_length_steps = ceil(episode_length_s / (decimation_rate * physics_time_step))

    For example, if the decimation rate is 10, the physics time step is 0.01, and the episode length is 10 seconds,
    then the episode length in steps is 100.
    """

    # environment settings
    rewards: object = None
    """Reward settings.

    Please refer to the :class:`isaaclab.managers.RewardManager` class for more details.
    """

    terminations: object = TerminationsCfg()
    """Termination settings.

    Please refer to the :class:`isaaclab.managers.TerminationManager` class for more details.
    """

    curriculum: object | None = None
    """Curriculum settings. Defaults to None, in which case no curriculum is applied.

    Please refer to the :class:`isaaclab.managers.CurriculumManager` class for more details.
    """

    commands: object | None = None
    """Command settings. Defaults to None, in which case no commands are generated.

    Please refer to the :class:`isaaclab.managers.CommandManager` class for more details.
    """

    rerender_on_reset: bool = False
    """Whether a render step is performed again after at least one environment has been reset.
    Defaults to False, which means no render step will be performed after reset.

    * When this is False, data collected from sensors after performing reset will be stale and will not reflect the
      latest states in simulation caused by the reset.
    * When this is True, an extra render step will be performed to update the sensor data
      to reflect the latest states from the reset. This comes at a cost of performance as an additional render
      step will be performed after each time an environment is reset.

    """

    num_rerenders_on_reset: int = 0

    wait_for_textures: bool = True
    """True to wait for assets to be loaded completely, False otherwise. Defaults to True."""

    export_io_descriptors: bool = False
    log_dir: str | None = None

    articulated_assets: dict[str, MatterixArticulationCfg] = {}

    objects: dict[str, MatterixRigidObjectCfg | MatterixStaticObjectCfg] = {}

    particle_systems: dict[str, ParticleSystemCfg] = {}

    reserved_particle_systems: list[ReservedParticleCfg] | None = None

    semantics: list[SemanticsCfg] | None = None  # global (environment level) semantics

    sensors: dict[str, SensorBaseCfg] = {}

    workflows: dict[str, object] = {}
    """Workflows available in the environment."""

    # general settings
    replicate_physics = False
    env_spacing = 2.5
    dt: float = 1 / 60  # Hz
    num_envs = 1
    export_io_descriptors = False

    # physX settings
    bounce_threshold_velocity = 0.01
    gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
    gpu_total_aggregate_pairs_capacity = 16 * 1024
    friction_correlation_distance = 0.00625

    def prepare_for_video_rec(
        self,
        fps: int = 60,
        resolution: tuple[int, int] = (3840, 2160),
        render: Literal["default", "balanced", "quality", "pathtracing"] = "pathtracing",
        eye: tuple[float, float, float] | None = None,
        lookat: tuple[float, float, float] | None = None,
    ) -> "MatterixBaseEnvCfg":
        """Configure viewer + sim render for video capture. Must be called before gym.make.

        Defaults to 60 fps, 4K path-traced. Presets mirror pretty-isaac:

            "default"      headless kit defaults: no translucency/reflections/GI/AO,
                           DLSS performance, 1 spp — fast, low quality
            "balanced"     bare ``RenderCfg(rendering_mode="balanced")``
            "quality"      rasterized with translucency, reflections, GI, AO,
                           shadows, DLAA, DL denoiser, direct lighting, 4 spp
            "pathtracing"  path-traced: DLAA + DL denoiser + spp=64,
                           totalSpp=512, maxBounces=6

        Returns self so calls can be chained.
        """
        import warnings

        physics_rate = round(1.0 / self.sim.dt)
        if fps > physics_rate:
            raise ValueError(
                f"Requested {fps} fps but physics runs at {physics_rate} Hz. "
                f"Either request fps <= {physics_rate} or decrease sim.dt "
                f"(currently {self.sim.dt}) to raise the physics rate."
            )

        render_interval = physics_rate // fps
        actual_fps = physics_rate // render_interval
        if actual_fps != fps:
            warnings.warn(
                f"Requested {fps} fps, but the minimum >= {fps} that evenly "
                f"divides {physics_rate} Hz is {actual_fps} fps "
                f"(render_interval={render_interval}). Using {actual_fps} fps."
            )

        self.sim.render_interval = render_interval
        self._recording_fps = actual_fps

        presets: dict[str, RenderCfg] = {
            "default": RenderCfg(),
            "balanced": RenderCfg(rendering_mode="balanced"),
            "quality": RenderCfg(
                rendering_mode="quality",
                enable_translucency=True,
                enable_reflections=True,
                enable_global_illumination=True,
                enable_ambient_occlusion=True,
                enable_shadows=True,
                enable_dl_denoiser=True,
                antialiasing_mode="DLAA",
                samples_per_pixel=4,
                enable_direct_lighting=True,
            ),
            "pathtracing": RenderCfg(
                rendering_mode="quality",
                antialiasing_mode="DLAA",
                enable_dl_denoiser=True,
                carb_settings={
                    "/rtx/rendermode": "PathTracing",
                    "/rtx/pathtracing/spp": 64,
                    "/rtx/pathtracing/totalSpp": 512,
                    "/rtx/pathtracing/maxBounces": 6,
                },
            ),
        }
        self.sim.render = presets[render]
        self.viewer.resolution = resolution
        if eye is not None:
            self.viewer.eye = eye
        if lookat is not None:
            self.viewer.lookat = lookat
        return self

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.enable_particles = False
        if self.particle_systems or self.reserved_particle_systems is not None:
            self.enable_particles = True

        if self.enable_particles:
            import omni.physx

            physx_interface = omni.physx.get_physx_interface()
            physx_interface.overwrite_gpu_setting(1)

            self.replicate_physics = True
            self.dt = 1 / 300  # change default physics time step from 60Hz to 300Hz when using particle systems
            self.sim.use_fabric = False
            # self.sim.device = "cpu" # this variable gets overwritten in scripts when loading the cfg file and arguments
            # so it is set in the base env class

        # simulation settings based on the user input
        if self.dt is not None:
            self.sim.dt = self.dt  # 100Hz
        self.sim.render_interval = self.decimation

        # physX settings
        if self.sim.physics is None:
            self.sim.physics = PhysxCfg()
        self.sim.physics.bounce_threshold_velocity = self.bounce_threshold_velocity
        self.sim.physics.gpu_found_lost_aggregate_pairs_capacity = self.gpu_found_lost_aggregate_pairs_capacity
        self.sim.physics.gpu_total_aggregate_pairs_capacity = self.gpu_total_aggregate_pairs_capacity
        self.sim.physics.friction_correlation_distance = self.friction_correlation_distance
