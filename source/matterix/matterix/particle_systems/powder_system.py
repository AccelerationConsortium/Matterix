# Copyright (c) 2022-2026, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Powder particle system."""

import time

import omni.kit.commands
import omni.timeline
from omni.physx.scripts import particleUtils
from pxr import PhysxSchema, Sdf, Usd, UsdGeom, Vt  # noqa: F401

from .particle_system import ParticleSystem
from .powder_cfg import PowderCfg

# =========================
# Tunables / “constants”
# =========================

# Solver
SOLVER_POSITION_ITERATIONS = 16

# Offsets & spacing (solid/powder)
SOLID_REST_OFFSET_COEFF = 0.99 * 0.6  # scales particle_contact_offset
PARTICLE_SPACING_MULT = 2.0  # particleSpacing = mult * restOffset
GRID_SAMPLE_PAD = 1  # +1 to include boundary

# Particle system limits
MAX_NEIGHBORHOOD = 96
MAX_VELOCITY = 200

# Anisotropy
ANISOTROPY_ENABLED = True
ANISOTROPY_SCALE = 2.5
ANISOTROPY_MIN_MULT = 0.3  # min = mult * scale
ANISOTROPY_MAX_MULT = 1.5  # max = mult * scale

# Smoothing
SMOOTHING_ENABLED = True
SMOOTHING_STRENGTH = 0.5

# Materials & paths
OPAQUE_MATERIAL_SUFFIX = "/OmniPBR"
TRANSPARENT_MATERIAL_SUFFIX = "/OmniSurfacePresets"

# Primvars / rendering
PRIMVAR_DO_NOT_CAST_SHADOWS_NAME = "doNotCastShadows"
PRIMVAR_DO_NOT_CAST_SHADOWS = True

# Point instancer / particle flags
ENABLE_SELF_COLLISION = True
ENABLE_FLUID = False  # powder/solid
PARTICLE_GROUP = 0
SPHERE_RADIUS_MULT = 1.0  # radius = mult * restOffset

# Timeline workaround
PAUSE_RESUME_WORKAROUND = True
PAUSE_SLEEP_SECS = 0.1


class PowderSystem(ParticleSystem):
    """PhysX particle-solid (powder) system with materials, anisotropy, and smoothing.

    Creates a PhysX particle system configured for solids/powders (fluid=False),
    sets materials and basic rendering flags, and instantiates a grid of particles.
    """

    def __init__(self, name: str, cfg: PowderCfg, env):
        """Initialize the powder system wrapper.

        Args:
            name: Unique name for this powder system instance.
            cfg:  PowderCfg with physical params (contact offset, density, etc.).
            env:  Environment/owner reference (passed to base class).
        """
        super().__init__(name, cfg, env)

    def create(self, stage, scenePath: Sdf.Path, prim_path: str):
        """Create and wire up the particle system, materials, and instancer.

        Args:
            stage:      USD stage to author into.
            scenePath:  Path to the physics scene prim (e.g., Sdf.Path("/physicsScene")).
            prim_path:  Base prim path under which this system will be created.
        """
        self.stage = stage
        self.scenePath = scenePath
        self.prim_path = prim_path

        # --- Particle system prim ------------------------------------------------
        self.particle_system_path = Sdf.Path(self.prim_path + f"/particleSystem_{self.name}")
        particle_system = PhysxSchema.PhysxParticleSystem.Define(stage, self.particle_system_path)
        particle_system.CreateSimulationOwnerRel().SetTargets([scenePath])

        # Rest offsets & spacing
        self.particleContactOffset = self.cfg.particle_contact_offset
        self.particleRestOffset = SOLID_REST_OFFSET_COEFF * self.particleContactOffset
        self.particleSpacing = PARTICLE_SPACING_MULT * self.particleRestOffset

        particle_system.CreateParticleContactOffsetAttr().Set(self.particleContactOffset)
        particle_system.CreateSolidRestOffsetAttr().Set(self.particleRestOffset)
        particle_system.CreateMaxNeighborhoodAttr().Set(MAX_NEIGHBORHOOD)
        particle_system.CreateMaxVelocityAttr().Set(MAX_VELOCITY)

        # --- Materials (opaque / transparent) -----------------------------------
        opaque_path = self.prim_path + OPAQUE_MATERIAL_SUFFIX
        transparent_path = self.prim_path + TRANSPARENT_MATERIAL_SUFFIX

        self.opaque_material_path = self.create_pbd_material(opaque_path, color_rgb=self.color_rgb, stage=self.stage)
        self.transparent_material_path = self.create_transparent_pbd_material(
            transparent_path, color_rgb=self.color_rgb, stage=self.stage
        )

        self.default_material_path = self.transparent_material_path if self.transparent else self.opaque_material_path
        self.material_prim_path = transparent_path if self.transparent else opaque_path

        # Apply PBD material parameters for powder (solid)
        particleUtils.add_pbd_particle_material(
            stage,
            self.default_material_path,
            cohesion=self.cfg.cohesion,
            viscosity=self.cfg.viscosity,
            surface_tension=self.cfg.surface_tension,
            friction=self.cfg.friction,
            cfl_coefficient=self.cfg.cfl_coefficient,
        )
        omni.kit.commands.execute(
            "BindMaterialCommand",
            prim_path=stage.GetPrimAtPath(self.particle_system_path).GetPath(),
            material_path=self.default_material_path,
            strength=None,
        )

        # --- Anisotropy ----------------------------------------------------------
        anisotropyAPI = PhysxSchema.PhysxParticleAnisotropyAPI.Apply(particle_system.GetPrim())
        anisotropyAPI.CreateParticleAnisotropyEnabledAttr().Set(ANISOTROPY_ENABLED)
        anisotropyAPI.CreateScaleAttr().Set(ANISOTROPY_SCALE)
        anisotropyAPI.CreateMinAttr().Set(ANISOTROPY_MIN_MULT * ANISOTROPY_SCALE)
        anisotropyAPI.CreateMaxAttr().Set(ANISOTROPY_MAX_MULT * ANISOTROPY_SCALE)

        # --- Smoothing -----------------------------------------------------------
        smoothingAPI = PhysxSchema.PhysxParticleSmoothingAPI.Apply(particle_system.GetPrim())
        smoothingAPI.CreateParticleSmoothingEnabledAttr().Set(SMOOTHING_ENABLED)
        smoothingAPI.CreateStrengthAttr().Set(SMOOTHING_STRENGTH)

        # --- Render/primvars -----------------------------------------------------
        primVarsApi = UsdGeom.PrimvarsAPI(particle_system)
        primVarsApi.CreatePrimvar(PRIMVAR_DO_NOT_CAST_SHADOWS_NAME, Sdf.ValueTypeNames.Bool).Set(
            PRIMVAR_DO_NOT_CAST_SHADOWS
        )

        # --- Particles grid & point instancer -----------------------------------
        self.particle_point_instancer_path = Sdf.Path(self.prim_path + "/particles")

        volume = self.cfg.volume
        lower = self.cfg.pos
        particleSpacing = self.particleSpacing
        num_samples = tuple(round(size / particleSpacing) + GRID_SAMPLE_PAD for size in volume)

        self.positions, self.velocities = particleUtils.create_particles_grid(
            lower, particleSpacing, num_samples[0], num_samples[1], num_samples[2]
        )

        self.instancer = particleUtils.add_physx_particleset_pointinstancer(
            stage,
            self.particle_point_instancer_path,
            Vt.Vec3fArray(self.positions),
            Vt.Vec3fArray(self.velocities),
            self.particle_system_path,
            self_collision=ENABLE_SELF_COLLISION,
            fluid=ENABLE_FLUID,  # powder/solid
            particle_group=PARTICLE_GROUP,
            particle_mass=self.cfg.particle_mass,
            density=self.cfg.density,
        )

        # Configure sphere prototype (visual glyph size)
        particle_prototype_sphere = UsdGeom.Sphere.Get(
            stage, self.particle_point_instancer_path.AppendChild("particlePrototype0")
        )
        particle_prototype_sphere.CreateRadiusAttr().Set(SPHERE_RADIUS_MULT * self.particleRestOffset)

        omni.kit.commands.execute(
            "BindMaterialCommand",
            prim_path=particle_prototype_sphere.GetPath(),
            material_path=self.default_material_path,
            strength=None,
        )

        # Refresh handle and solver iterations
        self.instancer = UsdGeom.PointInstancer.Get(self.stage, self.particle_point_instancer_path)
        particle_system.CreateSolverPositionIterationCountAttr().Set(SOLVER_POSITION_ITERATIONS)

        # Optional: pause/resume to ensure transparent material visibility
        if PAUSE_RESUME_WORKAROUND:
            timeline = omni.timeline.get_timeline_interface()
            timeline.pause()
            print("Simulation paused")
            time.sleep(PAUSE_SLEEP_SECS)
            timeline.play()
            print("Simulation resumed")
