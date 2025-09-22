# Copyright (c) 2022-2025, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Config classes for powder particle system"""

from dataclasses import MISSING

from isaaclab.utils import configclass

from .particle_cfg import ParticleSystemCfg, ReservedParticleCfg


@configclass
class PowderCfg(ParticleSystemCfg):
    pos: tuple[float, float, float] = MISSING  # unit: meter
    volume: tuple[float, float, float] = MISSING  # unit: meter
    # rendering
    color_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0)
    transparent: bool = False
    # physical properties
    particle_contact_offset: float = 0.0012  # in meters (this is their diameter)
    cohesion: float = 0.01
    viscosity: float = 20
    surface_tension: float = None
    friction: float = 0.5
    cfl_coefficient: float = 1.0
    density: float = 1000  # this is not matching the particle mass, the value is too high
    particle_mass: float = 0.000001


@configclass
class FinePowderCfg(ParticleSystemCfg):
    pos: tuple[float, float, float] = MISSING  # unit: meter
    volume: tuple[float, float, float] = MISSING  # unit: meter
    # rendering
    color_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0)
    transparent: bool = False
    # physical properties
    particle_contact_offset: float = 0.0005  # in meters (this is their diameter)
    cohesion: float = 0.01
    viscosity: float = 20
    surface_tension: float = None
    friction: float = 0.5
    cfl_coefficient: float = 1.0
    density: float = 1000  # this is not matching the particle mass, the value is too high
    particle_mass: float = 0.000001


@configclass
class ReservedPowderCfg(PowderCfg, ReservedParticleCfg):
    num_reserved_particle_sys: int = 0
    pos = (0.0, 0.0, -0.5)
    volume = (0.0, 0.0, 0.0)  # it will add the minimum one particle system
