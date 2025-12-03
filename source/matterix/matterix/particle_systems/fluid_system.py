# Copyright (c) 2022-2025, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Classes for fluid particle system"""

import time

import omni.kit.commands
import omni.timeline
from omni.physx.scripts import particleUtils
from pxr import PhysxSchema, Sdf, Usd, UsdGeom, Vt

from .fluid_cfg import FluidCfg
from .particle_system import ParticleSystem

# =========================
# Tunables / “constants”
# =========================

# Offsets & spacing
FLUID_REST_OFFSET_COEFF = 0.99 * 0.6  # scales particle_contact_offset
PARTICLE_SPACING_MULT = 2.0  # particleSpacing = mult * fluidRestOffset
GRID_SAMPLE_PAD = 1  # +1 in grid sampling to include boundary

# Particle system limits
PARTICLE_MAX_NEIGHBORHOOD = 300
PARTICLE_MAX_VELOCITY = 200
SOLVER_POSITION_ITERATION_COUNT = 32

# Anisotropy
ANISOTROPY_ENABLED = True
ANISOTROPY_SCALE = 2.5
ANISOTROPY_MIN_MULT = 0.3  # min = mult * scale
ANISOTROPY_MAX_MULT = 1.5  # max = mult * scale

# Smoothing
SMOOTHING_ENABLED = True
SMOOTHING_STRENGTH = 0.5

# Isosurface
ISOSURFACE_ENABLED = True
ISOSURFACE_MAX_VERTICES = 1024 * 1024
ISOSURFACE_MAX_TRIANGLES = 2 * 1024 * 1024
ISOSURFACE_MAX_SUBGRIDS = 1024 * 4
ISOSURFACE_GRID_SPACING_MULT = 0.9  # * fluidRestOffset
ISOSURFACE_SURFACE_DISTANCE_MULT = 0.95  # * fluidRestOffset
ISOSURFACE_GRID_FILTERING_PASSES = "GS"
ISOSURFACE_GRID_SMOOTHING_RADIUS_MULT = 1.0
ISOSURFACE_NUM_MESH_SMOOTHING_PASSES = 8
ISOSURFACE_NUM_MESH_NORMAL_SMOOTHING_PASSES = 8

# Materials & paths
OPAQUE_MATERIAL_SUFFIX = "/OmniPBR"
TRANSPARENT_MATERIAL_SUFFIX = "/OmniSurfacePresets"

# Primvars / rendering
PRIMVAR_DO_NOT_CAST_SHADOWS_NAME = "doNotCastShadows"
PRIMVAR_DO_NOT_CAST_SHADOWS = True

# Point instancer / particle flags
ENABLE_SELF_COLLISION = True
ENABLE_FLUID = True
PARTICLE_GROUP = 1
SPHERE_RADIUS_MULT = 1.0  # radius = mult * fluidRestOffset

# Timeline workaround
PAUSE_RESUME_WORKAROUND = True
PAUSE_SLEEP_SECS = 0.1


class FluidSystem(ParticleSystem):
    """PhysX particle-fluid system with isosurface rendering and materials.

    This wraps creation of a PhysX particle system, materials, anisotropy,
    smoothing, and isosurface settings, using values from `FluidCfg` and the
    tunables defined at the top of this file.
    """

    def __init__(self, name: str, cfg: FluidCfg, env):
        """Initialize the fluid system wrapper.

        Args:
            name: Unique name for this fluid system instance.
            cfg:  FluidCfg with physical params (contact offset, density, etc.).
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

        self.particleContactOffset = self.cfg.particle_contact_offset
        self.fluidRestOffset = FLUID_REST_OFFSET_COEFF * self.particleContactOffset
        self.particleSpacing = PARTICLE_SPACING_MULT * self.fluidRestOffset

        particle_system.CreateParticleContactOffsetAttr().Set(self.particleContactOffset)
        particle_system.CreateFluidRestOffsetAttr().Set(self.fluidRestOffset)
        particle_system.CreateMaxNeighborhoodAttr().Set(PARTICLE_MAX_NEIGHBORHOOD)
        particle_system.CreateMaxVelocityAttr().Set(PARTICLE_MAX_VELOCITY)

        # --- Materials (opaque / transparent) -----------------------------------
        opaque_path = self.prim_path + OPAQUE_MATERIAL_SUFFIX
        transparent_path = self.prim_path + TRANSPARENT_MATERIAL_SUFFIX
        self.opaque_material_path = self.create_pbd_material(opaque_path, color_rgb=self.color_rgb, stage=self.stage)
        self.transparent_material_path = self.create_transparent_pbd_material(
            transparent_path, color_rgb=self.color_rgb, stage=self.stage
        )

        # Choose default based on transparency flag from base class
        self.default_material_path = self.transparent_material_path if self.transparent else self.opaque_material_path
        self.material_prim_path = transparent_path if self.transparent else opaque_path

        # Apply PBD material parameters
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

        # --- Isosurface ----------------------------------------------------------
        isosurfaceAPI = PhysxSchema.PhysxParticleIsosurfaceAPI.Apply(particle_system.GetPrim())
        isosurfaceAPI.CreateIsosurfaceEnabledAttr().Set(ISOSURFACE_ENABLED)
        isosurfaceAPI.CreateMaxVerticesAttr().Set(ISOSURFACE_MAX_VERTICES)
        isosurfaceAPI.CreateMaxTrianglesAttr().Set(ISOSURFACE_MAX_TRIANGLES)
        isosurfaceAPI.CreateMaxSubgridsAttr().Set(ISOSURFACE_MAX_SUBGRIDS)
        isosurfaceAPI.CreateGridSpacingAttr().Set(self.fluidRestOffset * ISOSURFACE_GRID_SPACING_MULT)
        isosurfaceAPI.CreateSurfaceDistanceAttr().Set(self.fluidRestOffset * ISOSURFACE_SURFACE_DISTANCE_MULT)
        isosurfaceAPI.CreateGridFilteringPassesAttr().Set(ISOSURFACE_GRID_FILTERING_PASSES)
        isosurfaceAPI.CreateGridSmoothingRadiusAttr().Set(self.fluidRestOffset * ISOSURFACE_GRID_SMOOTHING_RADIUS_MULT)
        isosurfaceAPI.CreateNumMeshSmoothingPassesAttr().Set(ISOSURFACE_NUM_MESH_SMOOTHING_PASSES)
        isosurfaceAPI.CreateNumMeshNormalSmoothingPassesAttr().Set(ISOSURFACE_NUM_MESH_NORMAL_SMOOTHING_PASSES)

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
            fluid=ENABLE_FLUID,
            particle_group=PARTICLE_GROUP,
            particle_mass=self.cfg.particle_mass,
            density=self.cfg.density,
        )

        # Configure sphere prototype (radius drives on-screen glyph size)
        particle_prototype_sphere = UsdGeom.Sphere.Get(
            stage, self.particle_point_instancer_path.AppendChild("particlePrototype0")
        )
        particle_prototype_sphere.CreateRadiusAttr().Set(SPHERE_RADIUS_MULT * self.fluidRestOffset)

        omni.kit.commands.execute(
            "BindMaterialCommand",
            prim_path=particle_prototype_sphere.GetPath(),
            material_path=self.default_material_path,
            strength=None,
        )

        # Refresh handle and solver iterations
        self.instancer = UsdGeom.PointInstancer.Get(self.stage, self.particle_point_instancer_path)
        particle_system.CreateSolverPositionIterationCountAttr().Set(SOLVER_POSITION_ITERATION_COUNT)
        self.instancer.GetVisibilityAttr().Set("invisible")

        # Optional: pause/resume to ensure transparent material visibility
        if PAUSE_RESUME_WORKAROUND:
            timeline = omni.timeline.get_timeline_interface()
            timeline.pause()
            print("Simulation paused")
            time.sleep(PAUSE_SLEEP_SECS)
            timeline.play()
            print("Simulation resumed")
