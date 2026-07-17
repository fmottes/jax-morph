import jax
import jax.numpy as jnp
import numpy as np

import jax_morph as jxm


class GrowRate(jxm.SimulationStep):
    step_type = jxm.StepType.DYNAMIC

    def state_writes(self):
        return (jxm.RADIUS,)

    def __call__(self, state, *, dt, key):
        return state.deltas(radius=dt * jnp.ones_like(state.radius))


def test_simulate_integrates_to_time_and_is_differentiable():
    model = jxm.Model([GrowRate()])
    s0 = jxm.build_state_from_model(model).init_empty(capacity=2, n_space_dim=2, n_types=1)
    s0 = s0.set('alive', s0.alive.at[0].set(True))  # seed one live cell
    final = jxm.simulate(model, s0, n_steps=10, dt=0.1)  # deterministic: no key needed
    assert np.isclose(float(final.radius[0]), float(s0.radius[0]) + 1.0)  # 10 * 0.1 * 1
    assert np.isclose(float(final.t), 1.0)

    # pathwise-differentiable end to end: d final radius / d init radius = 1
    def loss(r0):
        s = s0.set('radius', s0.radius.at[0].set(r0))
        f = jxm.simulate(model, s, n_steps=10, dt=0.1)
        return f.radius[0]

    assert np.isclose(float(jax.grad(loss)(0.5)), 1.0)


def test_simulate_history_includes_initial_and_final_states():
    model = jxm.Model([GrowRate()])
    s0 = jxm.build_state_from_model(model).init_empty(capacity=2, n_space_dim=2, n_types=1)
    s0 = s0.set('alive', s0.alive.at[0].set(True))  # seed one live cell
    traj = jxm.simulate(model, s0, n_steps=5, dt=0.1, history=True)
    final = jxm.simulate(model, s0, n_steps=5, dt=0.1)
    assert traj.radius.shape == (6, 2)  # initial state plus five macro-steps
    assert np.allclose(np.asarray(traj.radius[0]), np.asarray(s0.radius))
    assert np.allclose(np.asarray(traj.radius[-1]), np.asarray(final.radius))
    assert np.isclose(float(traj.t[0]), float(s0.t))
    assert np.isclose(float(traj.t[-1]), 0.5)


def test_simulate_history_is_pathwise_differentiable():
    model = jxm.Model([GrowRate()])
    s0 = jxm.build_state_from_model(model).init_empty(capacity=2, n_space_dim=2, n_types=1)
    s0 = s0.set('alive', s0.alive.at[0].set(True))

    def loss(r0):
        state = s0.set('radius', s0.radius.at[0].set(r0))
        history = jxm.simulate(model, state, n_steps=5, dt=0.1, history=True)
        return history.radius[-1, 0]

    assert np.isclose(float(jax.grad(loss)(0.5)), 1.0)


def test_simulate_zero_step_history_contains_only_initial_state():
    model = jxm.Model([GrowRate()])
    s0 = jxm.build_state_from_model(model).init_empty(capacity=2, n_space_dim=2, n_types=1)
    s0 = s0.update(alive=s0.alive.at[0].set(True), t=jnp.asarray(2.5))
    traj = jxm.simulate(model, s0, n_steps=0, dt=0.1, history=True)
    assert traj.radius.shape == (1, 2)
    assert np.allclose(np.asarray(traj.radius[0]), np.asarray(s0.radius))
    assert np.isclose(float(traj.t[0]), 2.5)


def test_simulate_checkpoint_matches_non_checkpoint_in_value_and_gradient():
    # checkpoint=True rematerializes each macro-step in the backward pass; it must be a pure
    # memory/compute trade with identical value and gradient to the default path.
    model = jxm.Model([GrowRate()])
    s0 = jxm.build_state_from_model(model).init_empty(capacity=2, n_space_dim=2, n_types=1)
    s0 = s0.set('alive', s0.alive.at[0].set(True))

    def loss(r0, checkpoint):
        s = s0.set('radius', s0.radius.at[0].set(r0))
        f = jxm.simulate(model, s, n_steps=8, dt=0.1, checkpoint=checkpoint)
        return f.radius[0]

    assert np.isclose(float(loss(0.5, False)), float(loss(0.5, True)))  # same value
    g_off = float(jax.grad(lambda r: loss(r, False))(0.5))
    g_on = float(jax.grad(lambda r: loss(r, True))(0.5))
    assert np.isclose(g_off, g_on)  # same gradient
