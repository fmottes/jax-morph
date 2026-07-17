"""Tests for the SaturatingCellGrowth step (per-cell saturating radius dynamics).

SaturatingCellGrowth reads a per-cell ``growth_rate`` from state, so the rate can be a fixed initial
condition or an upstream controller's output. The tests cover the saturating approach to
``max_radius`` (including that the exact step stays stable where forward Euler would diverge), that
the rate is genuinely per-cell, dead-cell masking, and - the point of the state-field design - that
gradients flow both through the per-cell rate and back to an upstream controller's parameters.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import jax_morph as jxm
from jax_morph.physics.growth import GROWTH_RATE, SaturatingCellGrowth


def _seed(model, radii, rates, *, capacity=None):
    radii = jnp.asarray(radii, dtype=float)
    n = radii.shape[0]
    cap = capacity if capacity is not None else n
    State = jxm.build_state_from_model(model)
    s = State.init_empty(capacity=cap, n_space_dim=2, n_types=1)
    return s.update(
        alive=s.alive.at[:n].set(True),
        radius=s.radius.at[:n].set(radii),
        growth_rate=s.growth_rate.at[:n].set(jnp.asarray(rates, dtype=float)),
        celltype=s.celltype.at[:n, 0].set(1.0),
    )


class SetGrowthRate(jxm.SimulationStep):
    """Toy quasistatic controller: set every cell's growth_rate to softplus(w) from a param w.

    Stands in for a decision module; SaturatingCellGrowth reads the rate it writes, so a gradient
    through growth reaches w only because the re-run / rollout recomputes growth_rate through w.
    """

    step_type = jxm.StepType.QUASISTATIC
    w: jax.Array

    def state_writes(self):
        return (
            GROWTH_RATE,
        )  # the SAME exported spec SaturatingCellGrowth reads (merge-compatible)

    def __call__(self, state, *, dt, key):
        rate = jax.nn.softplus(self.w) * jnp.ones_like(state.radius)
        return state.set('growth_rate', rate * state.alive.astype(rate.dtype))


def test_growth_increases_radius_toward_max():
    model = jxm.Model([SaturatingCellGrowth(max_radius=1.0)])
    s = _seed(model, [0.2], [1.0], capacity=2)
    final = jxm.simulate(model, s, n_steps=100, dt=0.1)  # deterministic: no key needed
    r = float(final.radius[0])
    assert 0.2 < r < 1.0 and abs(r - 1.0) < 0.05  # approaches, does not exceed, max_radius


def test_growth_rate_is_per_cell():
    model = jxm.Model([SaturatingCellGrowth(max_radius=2.0)])
    s = _seed(model, [0.5, 0.5], [1.0, 0.2])  # same size, different rates
    final = jxm.simulate(model, s, n_steps=20, dt=0.05)
    r_fast, r_slow = float(final.radius[0]), float(final.radius[1])
    assert r_fast > r_slow > 0.5  # both grow, the higher-rate cell grows more


def test_growth_leaves_dead_cells_untouched():
    # Give the dead slots a large growth_rate: they would grow toward max_radius if the step did not
    # rely on the model zeroing dead cells' cell-scope deltas, so the assertion actually exercises
    # that masking (rather than passing trivially because a dead cell's default rate is 0).
    model = jxm.Model([SaturatingCellGrowth(max_radius=1.0)])
    s = _seed(model, [0.2], [1.0], capacity=3)  # only slot 0 alive
    s = s.update(
        growth_rate=s.growth_rate.at[1:].set(5.0)
    )  # dead slots: a rate that WOULD grow them
    final = jxm.simulate(model, s, n_steps=10, dt=0.1)
    assert np.allclose(np.asarray(final.radius[1:]), 0.0)  # masked: dead cells' deltas are zeroed


def test_growth_gradient_flows_through_per_cell_rate():
    # The state-field design: differentiate a rollout w.r.t. the per-cell growth_rate carried in the
    # initial state; a higher rate yields a larger final radius, so the gradient is positive.
    model = jxm.Model([SaturatingCellGrowth(max_radius=1.0)])
    State = jxm.build_state_from_model(model)

    def final_radius(rate0):
        s = State.init_empty(capacity=1, n_space_dim=2, n_types=1)
        s = s.update(
            alive=s.alive.at[0].set(True),
            radius=s.radius.at[0].set(0.2),
            growth_rate=s.growth_rate.at[0].set(rate0),
        )
        return jxm.simulate(model, s, n_steps=10, dt=0.1).radius[0]

    g = float(jax.grad(final_radius)(1.0))
    assert g > 0.0  # faster per-cell growth rate -> larger final radius


def test_growth_rate_from_upstream_controller_is_optimizable():
    # The payoff of a state-field rate: a controller writes growth_rate from a param w,
    # SaturatingCellGrowth reads it, and the gradient of a size objective reaches w through the
    # rollout (the whole model is the policy). A module-param rate could not be driven per-cell.
    model = jxm.Model([SetGrowthRate(w=jnp.array(0.0)), SaturatingCellGrowth(max_radius=1.0)])
    s = _seed(model, [0.2, 0.2, 0.2], [0.0, 0.0, 0.0])  # rates get overwritten by the controller

    def total_size(m):
        final = jxm.simulate(m, s, n_steps=10, dt=0.1)
        return (final.radius * final.alive).sum()

    g = eqx.filter_grad(total_size)(model)
    assert float(g.steps[0].w) > 0.0  # more growth rate -> more mass; grad reaches the controller


def test_max_radius_is_a_plain_field_and_optimizable():
    # max_radius as a jax.Array is traced: raising the target size lets cells grow larger, so the
    # gradient of the final radius w.r.t. max_radius is positive.
    s_model = jxm.Model([SaturatingCellGrowth(max_radius=jnp.array(1.0))])
    s = _seed(s_model, [0.5], [1.0], capacity=1)

    def final_r(max_r):
        m = jxm.Model([SaturatingCellGrowth(max_radius=max_r)])
        return jxm.simulate(m, s, n_steps=10, dt=0.1).radius[0]

    g = float(jax.grad(final_r)(1.0))
    assert g > 0.0  # a larger asymptote -> a larger radius after the same time


def test_growth_is_stable_for_large_dt_times_rate():
    # Forward Euler diverges once dt * growth_rate / max_radius > 2 (the radius oscillates and blows
    # up); the exact exponential step stays bounded in [r0, max_radius] and monotone for any dt*rate.
    model = jxm.Model([SaturatingCellGrowth(max_radius=1.0)])
    s = _seed(model, [0.2], [30.0], capacity=1)  # dt*rate/max_radius = 3 -> old Euler divergence
    final = jxm.simulate(model, s, n_steps=10, dt=0.1)
    r = float(final.radius[0])
    assert np.isfinite(r) and 0.9 < r < 1.0  # saturates to max_radius; no overshoot or divergence
