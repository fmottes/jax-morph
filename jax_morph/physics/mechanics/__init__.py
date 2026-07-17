"""Mechanics: interaction potentials, mechanical relaxation, Brownian dynamics, virial stress.

The mechanical-interaction subsystem. ``potentials`` is the base layer (the ``Potential`` protocol
and the concrete pair potentials); ``relaxation`` and ``dynamics`` move positions under a potential;
``stress`` exposes the potential's virial as a sensing field. This package re-exports every public
name, so ``from jax_morph.physics.mechanics import ...`` resolves regardless of the internal split.
"""

from .dynamics import ACTIVE_HEADING, ACTIVE_SPEED, ActiveBrownianDynamics2D, BrownianDynamics
from .potentials import (
    Harmonic,
    Hertzian,
    LennardJones,
    Morse,
    NoForce,
    PairwisePotential,
    Potential,
    SoftSphere,
)
from .relaxation import MechanicalRelaxation, relax_equilibrium
from .stress import VirialStress

__all__ = [
    'Potential',
    'NoForce',
    'PairwisePotential',
    'Morse',
    'SoftSphere',
    'Hertzian',
    'Harmonic',
    'LennardJones',
    'relax_equilibrium',
    'MechanicalRelaxation',
    'BrownianDynamics',
    'ActiveBrownianDynamics2D',
    'ACTIVE_SPEED',
    'ACTIVE_HEADING',
    'VirialStress',
]
