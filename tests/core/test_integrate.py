import jax.numpy as jnp
import numpy as np

from jax_morph.core.state import RADIUS, build_state_from_model
from jax_morph.core.step import Model, SimulationStep, StepType


class GrowRate(SimulationStep):
    step_type = StepType.DYNAMIC
    rate: float

    def state_writes(self):
        return (RADIUS,)

    def __call__(self, state, *, dt, key):
        # a dynamic step returns its own dt-scaled increment as a sparse delta state
        return state.deltas(radius=self.rate * dt * jnp.ones_like(state.radius))


def test_two_dynamic_writers_accumulate_and_step(key):
    model = Model([GrowRate(1.0), GrowRate(2.0)])
    s = build_state_from_model(model).init_empty(capacity=3, n_space_dim=2, n_types=1)
    s = s.set('alive', s.alive.at[0].set(True))  # seed one live cell (empty state otherwise)
    r0 = float(s.radius[0])
    s2 = model(s, dt=0.5, key=key)
    # dr = (1 + 2) * 0.5 = 1.5 on the live cell; the dead slot is untouched; t advances by dt
    assert np.isclose(float(s2.radius[0]), r0 + 1.5)
    assert np.isclose(float(s2.radius[1]), 0.0)  # dead cell untouched (alive-masked)
    assert np.isclose(float(s2.t), 0.5)
