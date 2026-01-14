# Copyright (c) 2022-2026, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Config classes for fluid particle system"""

from dataclasses import MISSING

from isaaclab.utils import configclass

from .particle_cfg import ParticleSystemCfg, ReservedParticleCfg


@configclass
class FluidCfg(ParticleSystemCfg):
    pos: tuple[float, float, float] = MISSING  # unit: meter, center of cuboid
    volume: tuple[float, float, float] = MISSING  # unit: meter, center of cuboid
    # rendering
    color_rgb: tuple[float, float, float] = (0.85, 0.92, 0.96)
    transparent: bool = True
    # physical properties
    particle_contact_offset: float = 0.004  # in meters
    cohesion: float = 0.01
    viscosity: float = 0.0091
    surface_tension: float = 0.074
    friction: float = 0.01
    cfl_coefficient: float = 1.0
    density: float = 1000  # this is not matching the particle mass, the value is too high
    particle_mass: float = 0.000001


# to have better visualization and more refined fluid
@configclass
class FineGrainedFluidCfg(FluidCfg):
    particle_mass = 0.0000001
    particle_contact_offset = 0.002


@configclass
class ReservedFluidCfg(FluidCfg, ReservedParticleCfg):
    num_reserved_particle_sys: int = 0
    pos = (0.0, 0.0, -0.5)
    volume = (0.0, 0.0, 0.0)  # it will add the minimum one particle system
