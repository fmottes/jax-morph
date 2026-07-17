"""jax-morph: differentiable particle-based physics for proliferating cells and active matter.

Installed usage guides are available through ``guides.guide()`` and ``guides.list_guides()``.
To implement custom simulation steps, read ``guides.guide('extending')``.
"""

from importlib.metadata import version

from . import control, guides, physics, viz
from .core import (
    ALIVE,
    CELLTYPE,
    POSITION,
    RADIUS,
    TIME,
    BaseState,
    Model,
    SimulationStep,
    StateFieldSpec,
    StepType,
    StochasticStep,
    TrajectoryRecord,
    ad_utils,
    build_state_from_model,
    check_stochastic_step,
    geometry,
    load_model,
    load_state,
    load_trajectory,
    save_model,
    save_state,
    save_trajectory,
    simulate,
    trajectory_logp,
    transition_logp,
)

__version__ = version('jax-morph')

__all__ = [
    '__version__',
    'BaseState',
    'StateFieldSpec',
    'build_state_from_model',
    'POSITION',
    'RADIUS',
    'CELLTYPE',
    'ALIVE',
    'TIME',
    'StepType',
    'Model',
    'SimulationStep',
    'StochasticStep',
    'check_stochastic_step',
    'save_model',
    'load_model',
    'save_state',
    'load_state',
    'save_trajectory',
    'load_trajectory',
    'TrajectoryRecord',
    'simulate',
    'trajectory_logp',
    'transition_logp',
    'ad_utils',
    'geometry',
    'guides',
    'physics',
    'control',
    'viz',
]
