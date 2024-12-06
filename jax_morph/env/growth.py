import jax
import jax.numpy as np

import equinox as eqx

from .._base import SimulationStep


class CellGrowth(SimulationStep):
    max_radius: float
    growth_rate: float
    growth_type: str = eqx.field(static=True)
    _smoothing_exp: float = eqx.field(static=True)

    def return_logprob(self) -> bool:
        return False

    def __init__(
        self,
        *,
        growth_rate=1.0,
        max_radius=0.5,
        growth_type="linear",
        smoothing_exp=10.0,
        **kwargs
    ):

        # if not hasattr(state, 'radius'):
        #     raise AttributeError('CellState must have "radius" attribute')

        if growth_type not in ["linear", "exponential"]:
            raise ValueError('growth_type must be either "linear" or "exponential"')

        self.growth_rate = growth_rate
        self.max_radius = max_radius
        self.growth_type = growth_type

        # lower values cause some overshoot
        # if smoothing_exp = 0, then no smoothing
        self._smoothing_exp = smoothing_exp

    # Define the smooth transition function
    def smooth_transition(self, x):

        smooth_step = lambda x, y, k: 1 / (1 + np.exp(-k * (x - y)))

        diff_step = smooth_step(x, self.max_radius, self._smoothing_exp)

        return x * (1 - diff_step) + self.max_radius * diff_step

    @jax.named_scope("jax_morph.CellGrowth")
    def __call__(self, state, *, key=None, **kwargs):

        if self.growth_type == "linear":
            new_radius = state.radius + self.growth_rate
        elif self.growth_type == "exponential":
            new_radius = state.radius * np.exp(self.growth_rate)

        no_smooth_fn = lambda new_radius: np.where(
            new_radius > self.max_radius, self.max_radius, new_radius
        ) * np.where(state.celltype.sum(1)[:, None] > 0, 1, 0)

        smooth_fn = lambda new_radius: self.smooth_transition(new_radius) * np.where(
            state.celltype.sum(1)[:, None] > 0, 1, 0
        )

        new_radius = jax.lax.cond(
            self._smoothing_exp > 0, smooth_fn, no_smooth_fn, new_radius
        )

        state = eqx.tree_at(lambda s: s.radius, state, new_radius)

        return state
