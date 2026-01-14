# Copyright (c) 2022-2026, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base particle system class"""

import torch

import omni.kit.commands
from omni.physx.scripts import particleUtils, physicsUtils
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt

from .particle_cfg import ParticleSystemCfg

# =========================
# Tunables / “constants”
# =========================

# Material library names
PBR_MDL_NAME = "OmniPBR.mdl"
PBR_MTL_NAME = "OmniPBR"
SURFACE_MDL_NAME = "OmniSurfacePresets.mdl"
SURFACE_MTL_NAME = "OmniSurface_ClearWater"

# Material prim path suffixes
OPAQUE_MATERIAL_SUFFIX = "/OmniPBR"
TRANSPARENT_MATERIAL_SUFFIX = "/OmniSurfacePresets"
SHADER_CHILD_NAME = "/Shader"

# Default look parameters
DEFAULT_EMISSION_INTENSITY: float = 500.0
DEFAULT_EMISSION_WEIGHT_TRANSPARENT: float = 0.5

# Rendering / USD
DEFAULT_INTERPOLATION = Usd.InterpolationTypeHeld


class ParticleSystem:
    """Base particle system utilities shared by fluid and powder implementations.

    Child classes are expected to:
      - implement `create(...)` and set:
          self.stage, self.scenePath, self.prim_path
          self.particle_system_path, self.particle_point_instancer_path
          self.positions, self.velocities
          self.material_prim_path, self.opaque_material_path, self.transparent_material_path
      - maintain `self.instancer` handle to the PointInstancer.
    """

    def __init__(self, name: str, cfg: ParticleSystemCfg, env):
        self.name = name
        self.cfg = cfg
        if cfg.color_rgb is None:
            raise ValueError("cfg.color_rgb must be provided")
        self.color_rgb = tuple(cfg.color_rgb)
        self.emission_intensity = 0.0
        self.transparent = bool(cfg.transparent)
        self.env = env
        self.systems = []  # optional registry for child use

        # Populated by child `create(...)`
        self.stage = None
        self.scenePath = None
        self.prim_path = None
        self.material_prim_path = None
        self.opaque_material_path = None
        self.transparent_material_path = None
        self.particle_system_path = None
        self.particle_point_instancer_path = None
        self.instancer = None
        self.positions = None
        self.velocities = None
        self.particleSpacing = None  # child sets this based on offsets

    # -------------------------
    # Material look controls
    # -------------------------

    def set_emission(self, enable_emission: bool = True, emission_intensity: float = DEFAULT_EMISSION_INTENSITY):
        """Enable/disable emission and set intensity on the bound material shader."""
        shader = UsdShade.Shader.Get(self.stage, self.material_prim_path + SHADER_CHILD_NAME)
        self.emission_intensity = float(emission_intensity if enable_emission else 0.0)

        if self.transparent:
            # OmniSurfacePresets.mdl
            shader.CreateInput("emission_intensity", Sdf.ValueTypeNames.Float).Set(self.emission_intensity)
            shader.CreateInput("emission_weight", Sdf.ValueTypeNames.Float).Set(DEFAULT_EMISSION_WEIGHT_TRANSPARENT)
        else:
            # OmniPBR.mdl
            shader.CreateInput("enable_emission", Sdf.ValueTypeNames.Bool).Set(bool(enable_emission))
            shader.CreateInput("emissive_intensity", Sdf.ValueTypeNames.Float).Set(self.emission_intensity)

    def get_emission(self) -> float:
        """Return current emission intensity setting."""
        return float(self.emission_intensity)

    def set_color(self, color_rgb: tuple[float, float, float] | None = None):
        """Set the material color (diffuse for PBR, transmission/reflect for transparent)."""
        if color_rgb is not None:
            self.color_rgb = tuple(color_rgb)

        shader = UsdShade.Shader.Get(self.stage, self.material_prim_path + SHADER_CHILD_NAME)
        if self.transparent:
            # OmniSurfacePresets.mdl
            shader.CreateInput("specular_transmission_color", Sdf.ValueTypeNames.Color3f).Set(self.color_rgb)
            shader.CreateInput("diffuse_reflection_color", Sdf.ValueTypeNames.Color3f).Set(self.color_rgb)
        else:
            # OmniPBR.mdl
            shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(self.color_rgb)

    def get_color(self) -> tuple[float, float, float]:
        """Return the current RGB color tuple."""
        return tuple(self.color_rgb)

    # -------------------------
    # Lifecycle / binding
    # -------------------------

    def create(self, stage, scenePath, prim_path):
        """Abstract: child classes must implement environment creation."""
        raise NotImplementedError

    def change_material_transparency(self, transparent: bool):
        """Switch between opaque and transparent material bindings in-place."""
        self.transparent = bool(transparent)

        if self.transparent and self.material_prim_path == self.prim_path + OPAQUE_MATERIAL_SUFFIX:
            self.material_prim_path = self.prim_path + TRANSPARENT_MATERIAL_SUFFIX
            omni.kit.commands.execute(
                "BindMaterialCommand",
                prim_path=self.stage.GetPrimAtPath(self.particle_system_path).GetPath(),
                material_path=self.transparent_material_path,
                strength=None,
            )

        if (not self.transparent) and self.material_prim_path == self.prim_path + TRANSPARENT_MATERIAL_SUFFIX:
            self.material_prim_path = self.prim_path + OPAQUE_MATERIAL_SUFFIX
            omni.kit.commands.execute(
                "BindMaterialCommand",
                prim_path=self.stage.GetPrimAtPath(self.particle_system_path).GetPath(),
                material_path=self.opaque_material_path,
                strength=None,
            )

    # -------------------------
    # State / transforms
    # -------------------------

    def set_pos(self, pos: tuple[float, float, float]):
        """Translate the point instancer to a world-space position."""
        point_instancer = UsdGeom.PointInstancer.Get(self.stage, self.particle_point_instancer_path)
        physicsUtils.set_or_add_translate_op(point_instancer, translate=pos)

    def reset(self):
        """Reset particles to initial positions/velocities and restore look."""
        instancer = UsdGeom.PointInstancer.Define(self.stage, self.particle_point_instancer_path)
        instancer.GetPositionsAttr().Set(Vt.Vec3fArray(self.positions))
        instancer.GetVelocitiesAttr().Set(Vt.Vec3fArray(self.velocities))
        self.set_emission(enable_emission=False, emission_intensity=0.0)
        self.set_color(self.cfg.color_rgb)

    # -------------------------
    # Particle editing helpers
    # -------------------------

    def add_particle_flow(
        self,
        particle_rate: tuple[int, int, int],
        pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
        offset: bool = False,
        vel: tuple[float, float, float] = (0.0, 0.0, -1.0),
    ):
        """Add a block of particles (nx, ny, nz) at `pos` with uniform initial velocity."""
        if offset:
            pos = (pos[0] + self.cfg.pos[0], pos[1] + self.cfg.pos[1], pos[2] + self.cfg.pos[2])

        instancer = UsdGeom.PointInstancer.Define(self.stage, self.particle_point_instancer_path)
        positions_attr = instancer.GetPositionsAttr()
        velocities_attr = instancer.GetVelocitiesAttr()

        current_positions = positions_attr.Get()
        current_velocities = velocities_attr.Get()

        positions, velocities = particleUtils.create_particles_grid(
            pos,
            self.particleSpacing,
            particle_rate[0],
            particle_rate[1],
            particle_rate[2],
            uniform_particle_velocity=Gf.Vec3f(*vel),
        )

        # Append new block
        current_positions = Vt.Vec3fArray(list(current_positions) + positions)
        current_velocities = Vt.Vec3fArray(list(current_velocities) + velocities)

        positions_attr.Set(current_positions)
        velocities_attr.Set(current_velocities)

    def remove_particle_flow(self, particle_rate: int):
        """Remove the last `particle_rate` particles (simple tail-pop helper)."""
        instancer = UsdGeom.PointInstancer.Get(self.stage, self.particle_point_instancer_path)
        positions_attr = instancer.GetPositionsAttr()
        velocities_attr = instancer.GetVelocitiesAttr()

        positions_list = list(positions_attr.Get())
        velocities_list = list(velocities_attr.Get())

        remove_n = min(max(int(particle_rate), 0), len(positions_list))
        if remove_n > 0:
            del positions_list[-remove_n:]
            del velocities_list[-remove_n:]

        positions_attr.Set(Vt.Vec3fArray(positions_list))
        velocities_attr.Set(Vt.Vec3fArray(velocities_list))

    # -------------------------
    # Geom queries & snapshots
    # -------------------------

    def find_object_bounding_box(self, obj_name: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (min, max) world-space AABB of a named object from env.objects."""
        print(self.env.objects[obj_name].objects.name)
        prim = self.env.objects[obj_name].objects.prims[0]

        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=["default"])
        world_bbox = bbox_cache.ComputeWorldBound(prim)
        range_3d = world_bbox.GetRange()
        min_point = range_3d.GetMin()
        max_point = range_3d.GetMax()

        device = getattr(self.env, "device", "cpu")
        cuboid_min_torch = torch.tensor([min_point[0], min_point[1], min_point[2]], dtype=torch.float32, device=device)
        cuboid_max_torch = torch.tensor([max_point[0], max_point[1], max_point[2]], dtype=torch.float32, device=device)
        return cuboid_min_torch, cuboid_max_torch

    def get_current_pos_vel(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return current particle positions and velocities as torch tensors on env.device."""
        positions_attr = self.instancer.GetPositionsAttr()
        velocities_attr = self.instancer.GetVelocitiesAttr()
        device = getattr(self.env, "device", "cpu")
        current_positions = torch.tensor(positions_attr.Get(), dtype=torch.float32, device=device)
        current_velocities = torch.tensor(velocities_attr.Get(), dtype=torch.float32, device=device)
        return current_positions, current_velocities

    # -------------------------
    # Material creation
    # -------------------------

    def create_pbd_material(self, target_path: str, color_rgb: tuple | None = None, stage=None) -> Sdf.Path:
        """Create/Move an OmniPBR material at `target_path` and set diffuse color."""
        stage = stage or self.stage
        if stage is None:
            raise ValueError("`stage` must be available to create_pbd_material")
        # use existing default color in cfg file if none provided
        if color_rgb is None:
            color_rgb = self.color_rgb

        created = []
        omni.kit.commands.execute(
            "CreateAndBindMdlMaterialFromLibrary",
            mdl_name=PBR_MDL_NAME,
            mtl_name=PBR_MTL_NAME,
            mtl_created_list=created,
            bind_selected_prims=False,
        )
        if created and created[0] != target_path:
            omni.kit.commands.execute("MovePrims", paths_to_move={created[0]: target_path})
        shader = UsdShade.Shader.Get(stage, target_path + SHADER_CHILD_NAME)
        shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(color_rgb)
        return Sdf.Path(target_path)

    def create_transparent_pbd_material(self, target_path: str, color_rgb: tuple | None = None, stage=None) -> Sdf.Path:
        """Create/Move an OmniSurface material at `target_path` and set color fields."""
        stage = stage or self.stage
        if stage is None:
            raise ValueError("`stage` must be available to create_transparent_pbd_material")
        # use existing default color in cfg file if none provided
        if color_rgb is None:
            color_rgb = self.color_rgb

        created = []
        omni.kit.commands.execute(
            "CreateAndBindMdlMaterialFromLibrary",
            mdl_name=SURFACE_MDL_NAME,
            mtl_name=SURFACE_MTL_NAME,
            mtl_created_list=created,
            bind_selected_prims=False,
        )
        if created and created[0] != target_path:
            omni.kit.commands.execute("MovePrims", paths_to_move={created[0]: target_path})
        shader = UsdShade.Shader.Get(stage, target_path + SHADER_CHILD_NAME)
        shader.CreateInput("specular_transmission_color", Sdf.ValueTypeNames.Color3f).Set(color_rgb)
        shader.CreateInput("diffuse_reflection_color", Sdf.ValueTypeNames.Color3f).Set(color_rgb)
        shader.CreateInput("thin_walled", Sdf.ValueTypeNames.Bool).Set(True)

        return Sdf.Path(target_path)
