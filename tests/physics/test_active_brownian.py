"""Tests for ActiveBrownianDynamics2D: self-propelled overdamped Langevin dynamics.

Validated against the free-drift limit (pure self-propulsion, no force/noise), the passive limit
(v0 = 0 reduces to no motion when noise/force are off), the heading's rotational-diffusion variance,
the stochastic-step trace round-trip contract, and both estimators' gradients (score-function into
active_speed, pathwise into rot_diffusion).
"""

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

import jax_morph as jxm
from jax_morph.core.step import check_stochastic_step
from jax_morph.physics.mechanics import ActiveBrownianDynamics2D, NoForce, SoftSphere


def _abp(n_seed, *, capacity=None, v0=1.0, theta0=0.0, potential=None, **kw):
    capacity = capacity or n_seed
    step = ActiveBrownianDynamics2D(potential or SoftSphere(), n_space_dim=2, **kw)
    s = jxm.build_state_from_model(step).init_empty(capacity=capacity, n_space_dim=2, n_types=1)
    s = s.update(
        alive=s.alive.at[:n_seed].set(True),
        radius=s.radius.at[:n_seed].set(0.5),
        active_speed=s.active_speed.at[:n_seed].set(v0),
        active_heading=s.active_heading.at[:n_seed].set(theta0),
    )
    return step, s


def test_free_cell_drifts_along_its_heading(key):
    # one isolated cell (no neighbours -> zero force), no noise: dx = dt * v0 * e(theta).
    step, s = _abp(1, v0=2.0, theta0=jnp.pi / 2, kT=0.0, rot_diffusion=0.0)  # heading +y
    s2 = jxm.Model([step])(s, dt=0.5, key=key)
    assert np.allclose(np.asarray(s2.position[0] - s.position[0]), [0.0, 0.5 * 2.0], atol=1e-6)
    assert np.isclose(
        float(s2.active_heading[0]), float(s.active_heading[0])
    )  # D_r=0 -> heading fixed


def test_zero_speed_is_passive(key):
    # v0 = 0: an isolated cell with no noise does not move (no active drift, force = 0).
    step, s = _abp(1, v0=0.0, kT=0.0, rot_diffusion=0.0)
    s2 = jxm.Model([step])(s, dt=1.0, key=key)
    assert np.allclose(np.asarray(s2.position[0] - s.position[0]), 0.0, atol=1e-9)


def test_heading_rotational_diffusion(key):
    # coincident cells, v0 = 0, and a zero-force potential (avoids a dense O(N^2) pairwise
    # computation at N=20000): the heading increment has var 2 D_r dt regardless of position.
    Dr, dt, N = 0.7, 0.3, 20000
    step, s = _abp(N, v0=0.0, theta0=0.0, kT=0.0, rot_diffusion=Dr, potential=NoForce())
    s2 = jxm.Model([step])(s, dt=dt, key=key)
    dtheta = np.asarray(s2.active_heading)[:N]  # theta0 = 0
    assert np.isclose(dtheta.var(), 2 * Dr * dt, rtol=0.05)
    assert np.all(np.isfinite(np.asarray(s2.position)))  # no NaN from coincident cells


def test_trace_round_trips(key):
    step, s = _abp(4, capacity=8, v0=1.0, kT=0.1, rot_diffusion=0.5)
    check_stochastic_step(step, s, dt=1.0, key=key)  # co-emits xi_t / dx / xi_r / dtheta


def test_logp_gradient_reaches_active_speed(key):
    # score-function gradient of the step logp wrt the per-cell speed is nonzero and finite.
    step, s = _abp(6, v0=1.0, kT=0.1, rot_diffusion=0.5)
    s2 = jxm.Model([step])(s, dt=1.0, key=key)
    trace = step.trace_from_state(s2)
    g = eqx.filter_grad(lambda v0: step.logp(s.set('active_speed', v0), trace, 1.0))(s.active_speed)
    assert float(jnp.sum(jnp.abs(g))) > 0.0


def test_pathwise_gradient_reaches_the_rotational_rate(key):
    # grad(final heading spread) wrt rot_diffusion flows through the reparameterized heading increment.
    def spread(Dr):
        step = ActiveBrownianDynamics2D(SoftSphere(), n_space_dim=2, kT=0.0, rot_diffusion=Dr)
        s = jxm.build_state_from_model(step).init_empty(capacity=64, n_space_dim=2, n_types=1)
        s = s.update(alive=s.alive.at[:64].set(True), radius=s.radius.at[:64].set(0.5))
        f = jxm.simulate(jxm.Model([step]), s, n_steps=3, dt=0.5, key=key)
        return jnp.var(f.active_heading[:64])

    assert float(jnp.abs(eqx.filter_grad(spread)(jnp.array(0.5)))) > 0.0


def test_logp_is_finite_at_zero_translational_noise(key):
    # kT=0 (pure self-propulsion + heading noise) is the textbook ABP; the score must be finite,
    # scoring only the rotational noise (the deterministic translation contributes 0), not NaN.
    Dr, dt, n = 0.5, 1.0, 4
    step, s = _abp(n, capacity=6, v0=1.0, kT=0.0, rot_diffusion=Dr)
    model = jxm.Model([step])
    s2 = model(s, dt=dt, key=key)
    lp = float(jxm.transition_logp(model, s, s2, dt=dt, score='all'))
    assert np.isfinite(lp)
    std_r = np.sqrt(2 * Dr * dt)
    dtheta = np.asarray(s2.active_dtheta)[:n]
    ref = np.sum(-0.5 * (dtheta / std_r) ** 2 - np.log(std_r) - 0.5 * np.log(2 * np.pi))
    assert np.isclose(lp, ref, atol=1e-5)  # only the heading noise is scored


def test_logp_is_finite_at_zero_rotational_noise(key):
    # D_r=0 (fixed heading, only translational noise): the score is finite (heading contributes 0).
    step, s = _abp(4, capacity=6, v0=1.0, kT=0.1, rot_diffusion=0.0)
    model = jxm.Model([step])
    s2 = model(s, dt=1.0, key=key)
    assert np.isfinite(float(jxm.transition_logp(model, s, s2, dt=1.0, score='all')))


def test_force_term_enters_the_drift(key):
    # two overlapping cells with no active drift and no noise separate by dt * force / gamma,
    # exercising the potential-force drift term (every other test sits at zero force).
    step = ActiveBrownianDynamics2D(
        SoftSphere(epsilon=1.0), n_space_dim=2, kT=0.0, rot_diffusion=0.0, gamma=2.0
    )
    s = jxm.build_state_from_model(step).init_empty(capacity=2, n_space_dim=2, n_types=1)
    s = s.update(
        alive=s.alive.at[:2].set(True),
        radius=s.radius.at[:2].set(0.5),
        position=s.position.at[:2].set(
            jnp.array([[0.0, 0.0], [0.6, 0.0]])
        ),  # overlap (dist < sigma=1)
    )
    s2 = jxm.Model([step])(s, dt=0.1, key=key)
    before = float(jnp.linalg.norm(s.position[0] - s.position[1]))
    after = float(jnp.linalg.norm(s2.position[0] - s2.position[1]))
    assert after > before  # repulsion pushed them apart along the force


def test_requires_two_dimensions():
    with pytest.raises(ValueError, match='2-D only'):
        ActiveBrownianDynamics2D(SoftSphere(), n_space_dim=3)


def test_potential_none_defaults_to_no_force(key):
    # potential=None -> NoForce: a free active gas (self-propulsion + heading noise, no interaction).
    # Two overlapping cells with v0=0 and no noise feel no force, so nothing moves - unlike SoftSphere,
    # which would push them apart (test_force_term_enters_the_drift).
    step = ActiveBrownianDynamics2D(None, n_space_dim=2, kT=0.0, rot_diffusion=0.0)
    assert isinstance(step.potential, NoForce)
    _, s = _abp(2, v0=0.0, kT=0.0, rot_diffusion=0.0, potential=NoForce())
    s = s.update(position=s.position.at[:2].set(jnp.array([[0.0, 0.0], [0.6, 0.0]])))  # overlap
    s2 = jxm.Model([step])(s, dt=0.1, key=key)
    assert np.allclose(np.asarray(s2.position), np.asarray(s.position))
