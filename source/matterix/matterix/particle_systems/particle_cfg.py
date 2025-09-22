# Copyright (c) 2022-2025, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Config classes for particle system"""

from dataclasses import MISSING

from isaaclab.utils import configclass


@configclass
class ParticleSystemCfg:
    pos: tuple[float, float, float] = MISSING  # unit: meter, center of cuboid
    volume: tuple[float, float, float] = MISSING  # unit: meter
    # Default rendering and look parameters
    color_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0)
    transparent: bool = False

    # physical properties
    particle_contact_offset: float = 0.0005  # in meters (this is their diameter)
    cohesion: float = 0.01
    viscosity: float = 20
    surface_tension: float | None = None
    friction: float = 0.5
    cfl_coefficient: float = 1.0
    density: float = (
        1000  # this is not matching the particle mass, the value is too high, set for better simulation behavior
    )
    particle_mass: float = 0.000001

    def __post_init__(self):
        # offset from the center of cuboid to the corner of the cuboid (particle system origin)
        if self.pos is MISSING:
            raise ValueError("ParticleSystemCfg: 'pos' must be specified.")
        if self.volume is MISSING:
            raise ValueError("ParticleSystemCfg: 'volume' must be specified.")
        self.pos = tuple(a - b / 2.0 for a, b in zip(self.pos, self.volume))


@configclass
class ReservedParticleCfg(ParticleSystemCfg):
    num_reserved_particle_sys: int = 0
    volume = (0.0, 0.0, 0.0)  # zero-volume will make a particle system with the minimum one particle
    pos = (0.0, 0.0, -0.5)
