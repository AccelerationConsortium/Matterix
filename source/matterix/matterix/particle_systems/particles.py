# Copyright (c) 2022-2026, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager for per-env particle systems (fluid or powder)."""

import copy
import torch

from pxr import Sdf

from .fluid_cfg import FluidCfg
from .fluid_system import FluidSystem
from .powder_cfg import PowderCfg
from .powder_system import PowderSystem

# check for material ids here:
# https://docs.omniverse.nvidia.com/materials-and-rendering/latest/materials.html#omnisurfacebase-emission


class Particles:
    """Manager for per-env particle systems (fluid or powder).

    This class creates one particle system per environment index and exposes
    explicit wrapper methods (e.g., `set_color`, `reset`) that fan out to
    selected envs by index. Use `env_ids` as:
      - `None`  -> all envs
      - `int`   -> single env
      - `slice` -> range of envs (e.g., `slice(0, 3)`)
      - `list`/`tuple` of ints -> specific envs
    """

    def __init__(self, name, cfg: FluidCfg | PowderCfg, env):
        """Initialize the particle systems manager.

        Args:
            name: Base name used to label per-env systems.
            cfg:  FluidCfg or PowderCfg used as a template for all envs.
            env:  Simulation environment providing `num_envs`, `scene`, and `device`.
        """
        self.env = env
        self.name = name
        self.cfg = cfg

        # One system per env (FluidSystem or PowderSystem).
        self.particle_systems: list[FluidSystem | PowderSystem] = []

        # Create and register per-env systems.
        for env_idx in range(env.num_envs):
            self.spawn_particles_for_env_idx(env_idx)

    def spawn_particles_for_env_idx(self, env_idx):
        """Create and initialize a particle system for a single environment index.

        Notes:
            - Keeps attribute names and behavior unchanged per your request.
            - Clones `self.cfg` and offsets its `pos` by the env origin.
        """
        # move these stuff to the interactive scene definition, so that it can manage it
        self.scenePath = Sdf.Path("/physicsScene")

        env_origin = self.env.scene.env_origins[env_idx]
        # we are modifying the same object, should clone it prob
        particle_cfg = copy.deepcopy(self.cfg)
        particle_cfg.pos = (torch.tensor(particle_cfg.pos, device=self.env.device) + env_origin).tolist()

        if "fluid" in particle_cfg.__class__.__name__.lower():
            self.particle_systems.append(FluidSystem(name=f"{self.name}_{env_idx}", cfg=particle_cfg, env=self.env))
            self.particle_systems[-1].create(
                self.env.scene.stage, self.scenePath, f"/World/Fluid/env_{env_idx}/{self.name}_{env_idx}"
            )
        else:
            self.particle_systems.append(PowderSystem(name=f"{self.name}_{env_idx}", cfg=particle_cfg, env=self.env))
            self.particle_systems[-1].create(
                self.env.scene.stage, self.scenePath, f"/World/Powder/env_{env_idx}/{self.name}_{env_idx}"
            )

        print(f"[INFO] Initializing particle system: {self.name}_{env_idx}.")

    # -------------------
    # explicit wrappers
    # -------------------
    def _targets_from_env_ids(self, env_ids):
        """Resolve `env_ids` into a list of target systems.

        Returns:
            (indices, targets): a pair where `indices` is the list of env indices
            and `targets` is the corresponding list of particle systems.

        Implementation detail:
            If env_ids is None, follow your convention:
            `env_ids = slice(env_ids)` which is `slice(None)` to select all.
        """
        if env_ids is None:
            env_ids = slice(env_ids)  # == slice(None)
        if isinstance(env_ids, slice):
            idxs = range(len(self.particle_systems))[env_ids]
            targets = [self.particle_systems[i] for i in idxs]
            return list(idxs), targets
        if isinstance(env_ids, int):
            return [env_ids], [self.particle_systems[env_ids]]
        # assume iterable of ints
        idxs = list(env_ids)
        targets = [self.particle_systems[i] for i in idxs]
        return idxs, targets

    def reset(self, env_ids=None) -> None:
        """Reset particles to their initial state for selected envs."""
        _, targets = self._targets_from_env_ids(env_ids)
        for sys in targets:
            sys.reset()

    def set_color(self, env_ids, color_rgb: tuple[float, float, float]) -> None:
        """Set display color on selected envs’ materials."""
        _, targets = self._targets_from_env_ids(env_ids)
        for sys in targets:
            sys.set_color(color_rgb)

    def get_color(self, env_ids=None) -> dict[int, tuple[float, float, float]]:
        """Get current display color for selected envs.

        Returns:
            Dict[env_idx, (r, g, b)]
        """
        idxs, targets = self._targets_from_env_ids(env_ids)
        return {i: sys.get_color() for i, sys in zip(idxs, targets)}

    def set_emission(self, env_ids, enable_emission: bool = True, emission_intensity: float = 500.0) -> None:
        """Enable/disable emission and set intensity on selected envs’ materials."""
        _, targets = self._targets_from_env_ids(env_ids)
        for sys in targets:
            sys.set_emission(enable_emission=enable_emission, emission_intensity=emission_intensity)

    def get_emission(self, env_ids=None) -> dict[int, float]:
        """Get current emission intensity for selected envs.

        Returns:
            Dict[env_idx, intensity]
        """
        idxs, targets = self._targets_from_env_ids(env_ids)
        return {i: sys.get_emission() for i, sys in zip(idxs, targets)}

    def change_material_transparency(self, env_ids, transparent: bool) -> None:
        """Switch materials between opaque and transparent for selected envs."""
        _, targets = self._targets_from_env_ids(env_ids)
        for sys in targets:
            sys.change_material_transparency(transparent)

    def set_pos(self, env_ids, pos: tuple[float, float, float]) -> None:
        """Translate the point instancer for selected envs to a world-space position."""
        _, targets = self._targets_from_env_ids(env_ids)
        for sys in targets:
            sys.set_pos(pos)

    def add_particle_flow(
        self,
        env_ids,
        particle_rate: tuple[int, int, int],
        pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
        offset: bool = False,
        vel: tuple[float, float, float] = (0.0, 0.0, -1.0),
    ) -> None:
        """Add a block of particles to selected envs.

        Args:
            particle_rate: (nx, ny, nz) counts for the grid to add.
            pos:           World-space spawn position (or offset if `offset=True`).
            offset:        Add `pos` to the system’s configured base position if True.
            vel:           Initial velocity assigned to the new particles.
        """
        _, targets = self._targets_from_env_ids(env_ids)
        for sys in targets:
            sys.add_particle_flow(particle_rate, pos=pos, offset=offset, vel=vel)

    def remove_particle_flow(self, env_ids, particle_rate: int) -> None:
        """Remove the last `particle_rate` particles from selected envs."""
        _, targets = self._targets_from_env_ids(env_ids)
        for sys in targets:
            sys.remove_particle_flow(particle_rate)

    def find_object_bounding_box(self, env_ids, obj_name: str):
        """Compute world-space AABB of an object for selected envs.

        Returns:
            Dict[env_idx, (min_tensor, max_tensor)]
        """
        idxs, targets = self._targets_from_env_ids(env_ids)
        return {i: sys.find_object_bounding_box(obj_name) for i, sys in zip(idxs, targets)}

    def get_current_pos_vel(self, env_ids=None):
        """Fetch current particle positions and velocities for selected envs.

        Returns:
            Dict[env_idx, (positions_tensor, velocities_tensor)]
        """
        idxs, targets = self._targets_from_env_ids(env_ids)
        return {i: sys.get_current_pos_vel() for i, sys in zip(idxs, targets)}

    def create_pbd_material(self, env_ids, target_path: str, color_rgb: tuple[float, float, float], stage=None) -> None:
        """Create/move a PBR material at `target_path` for selected envs and set diffuse color."""
        _, targets = self._targets_from_env_ids(env_ids)
        for sys in targets:
            sys.create_pbd_material(target_path, color_rgb=color_rgb, stage=stage)

    def create_transparent_pbd_material(
        self, env_ids, target_path: str, color_rgb: tuple[float, float, float], stage=None
    ) -> None:
        """Create/move a transparent OmniSurface material at `target_path` for selected envs and set colors."""
        _, targets = self._targets_from_env_ids(env_ids)
        for sys in targets:
            sys.create_transparent_pbd_material(target_path, color_rgb=color_rgb, stage=stage)
