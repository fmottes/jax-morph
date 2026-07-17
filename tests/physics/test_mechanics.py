"""Tests for the mechanics module: Morse potential, relaxation, and Brownian dynamics.

Morse is validated against its analytic well (zero force at contact, forces == -grad energy) and its
virial sign; the relaxation against a genuine force tolerance and its implicit-diff equilibrium
sensitivity; the Brownian step against the stochastic-step contract (trace round-trip, reproducible
draw) and its Gaussian ``logp`` (both the score-function gradient into its own parameters and the
pathwise gradient through ``simulate``).
"""

import warnings

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jax_morph as jxm
from jax_morph.core.state import build_state_from_model
from jax_morph.core.step import check_stochastic_step
from jax_morph.physics.mechanics import (
    BrownianDynamics,
    MechanicalRelaxation,
    Morse,
    NoForce,
    relax_equilibrium,
)


def _central_diff(f, x, eps=1e-6):
    """Central finite difference of a scalar function of a scalar."""
    return (f(x + eps) - f(x - eps)) / (2 * eps)


class _M:  # bare model: no declared fields, so the state carries only the base fields
    def state_requires(self):
        return ()


def _two_cells(dist, radius=0.5, n_types=1):
    s = build_state_from_model(_M()).init_empty(capacity=2, n_space_dim=2, n_types=n_types)
    pos = jnp.array([[0.0, 0.0], [dist, 0.0]])
    return s.update(
        alive=s.alive.at[:2].set(True),
        radius=s.radius.at[:2].set(radius),
        position=pos,
        celltype=s.celltype.at[:2, 0].set(1.0),  # both type 0 (one-hot)
    )


# ---------------------------------------------------------------------------
# NoForce (the trivial potential) and the potential=None default.
# ---------------------------------------------------------------------------


def test_no_force_is_zero_energy_and_force():
    # NoForce feels no interaction: zero energy and zero force even for an overlapping pair that any
    # real potential would push apart, and it reads no state fields.
    pot = NoForce()
    s = _two_cells(dist=0.6)  # heavily overlapping (sigma = 1.0)
    assert float(pot.total_energy(s.position, s)) == 0.0
    assert np.array_equal(np.asarray(pot.forces(s)), np.zeros_like(np.asarray(s.position)))
    assert pot.state_reads() == ()


# ---------------------------------------------------------------------------
# Morse potential.
# ---------------------------------------------------------------------------


def test_morse_minimum_and_force_at_contact():
    pot = Morse()
    s = _two_cells(dist=1.0)  # sigma = 0.5 + 0.5 = 1.0 -> at the well minimum
    f = pot.forces(s)
    assert np.allclose(np.asarray(f), 0.0, atol=1e-6)  # zero net force at equilibrium
    e_min = pot.total_energy(s.position, s)
    e_near = pot.total_energy(_two_cells(0.8).position, _two_cells(0.8))
    assert float(e_min) < float(e_near)  # contact is lower than compressed


def test_forces_match_autodiff_of_energy():
    pot = Morse()
    s = _two_cells(dist=0.8)
    g = jax.grad(lambda p: pot.total_energy(p, s))(s.position)
    assert np.allclose(np.asarray(pot.forces(s)), np.asarray(-g))


def test_morse_energy_ignores_dead_and_self_pairs():
    # A single live cell has no partner -> zero interaction energy and zero force (self-pairs and
    # the dead slot are masked out by neighbor_sum).
    pot = Morse()
    s = _two_cells(dist=1.2)
    s = s.set('alive', s.alive.at[1].set(False))  # kill the partner
    assert np.isclose(float(pot.total_energy(s.position, s)), 0.0)
    assert np.allclose(np.asarray(pot.forces(s)), 0.0)


def test_morse_scalar_epsilon_unchanged():
    # A shared scalar epsilon is used by every pair: the contact well is exactly -epsilon at r=sigma.
    s = _two_cells(dist=1.0)  # sigma = 1.0 -> at contact
    assert np.isclose(float(Morse(epsilon=2.5).total_energy(s.position, s)), -2.5)


def test_morse_coupling_mix_and_state_reads():
    # A scalar coupling sources no state field; a StateFieldSpec coupling declares its field, and the
    # per-cell -> per-pair rule is the arithmetic mean.
    assert Morse().state_reads() == ()
    spec = jxm.StateFieldSpec('adhesion')
    assert Morse(epsilon=spec).state_reads() == (spec,)
    v = jnp.array([1.0, 3.0, 5.0])
    assert np.allclose(np.asarray(Morse().mix(v)), 0.5 * (v[:, None] + v[None, :]))


def test_morse_per_cell_epsilon_from_state():
    # A per-cell epsilon lives in the state: the owning step surfaces the read so the field is
    # synthesized (no AttributeError), the pair coupling is the arithmetic mean of the two cells'
    # values, and the energy's gradient reaches state.adhesion (so a controller writing it is trainable).
    adhesion = jxm.StateFieldSpec('adhesion')
    pot = Morse(epsilon=adhesion)
    model = jxm.Model([MechanicalRelaxation(pot, max_steps=1, f_tol=1e-6)])
    s = build_state_from_model(model).init_empty(capacity=2, n_space_dim=2, n_types=1)
    s = s.update(
        alive=s.alive.at[:2].set(True),
        radius=s.radius.at[:2].set(0.5),
        position=jnp.array([[0.0, 0.0], [1.0, 0.0]]),
        adhesion=s.adhesion.at[:2].set(jnp.array([2.0, 4.0])),  # per-cell well depths
    )
    # at contact (r = sigma = 1.0) the pair energy is -epsilon_ij = -(2 + 4) / 2 = -3
    assert np.isclose(float(pot.total_energy(s.position, s)), -3.0)

    g = jax.grad(lambda adh: pot.total_energy(s.position, s.set('adhesion', adh)))(s.adhesion)
    assert np.all(np.isfinite(np.asarray(g))) and float(jnp.sum(jnp.abs(g))) > 0.0


def test_morse_beyond_cutoff_is_zero():
    pot = Morse()
    far = _two_cells(dist=3.0)  # r = 3 > r_cutoff_frac * sigma = 2.5 -> switched fully off
    assert np.isclose(float(pot.total_energy(far.position, far)), 0.0)
    assert np.allclose(np.asarray(pot.forces(far)), 0.0)


def test_virial_pressure_positive_under_compression_zero_apart():
    pot = Morse()
    compressed = pot.virial_pressure(_two_cells(0.6))  # r < sigma=1.0 -> repulsive -> p > 0
    apart = pot.virial_pressure(_two_cells(3.0))  # beyond cutoff -> ~0
    assert float(compressed[0]) > 0.0 and float(compressed[1]) > 0.0
    assert np.allclose(np.asarray(apart), 0.0, atol=1e-6)


def test_virial_pressure_optimizable_through_epsilon():
    # Compression pressure scales with the well depth epsilon, so the gradient reaches a traced eps.
    s = _two_cells(0.6)
    grad = jax.grad(lambda e: Morse(epsilon=e).virial_pressure(s)[0])(jnp.array(3.0))
    assert np.isfinite(float(grad)) and float(grad) != 0.0


def test_virial_pressure_uses_the_1d_cell_volume_in_one_dimension():
    # In 1D the cell 'volume' V_i is the interval length 2r, not the 3D sphere volume. Reconstruct
    # p_i = -(1 / (2 d V_i)) sum_j r_ij U'(r_ij) with d = 1, V_i = 2r and check the branch is used.
    pot = Morse()
    s = build_state_from_model(_M()).init_empty(capacity=2, n_space_dim=1, n_types=1)
    s = s.update(
        alive=s.alive.at[:2].set(True),
        radius=s.radius.at[:2].set(0.5),
        position=jnp.array([[0.0], [0.6]]),  # one neighbour at r = 0.6, sigma = 1.0
        celltype=s.celltype.at[:2, 0].set(1.0),
    )
    sigma, eps, alpha = (a[0, 1] for a in pot.pair_params(s))
    uprime = float(jax.grad(pot.pair_energy, argnums=0)(jnp.array(0.6), sigma, eps, alpha))
    expected = -(1.0 / (2 * 1 * (2 * 0.5))) * (0.6 * uprime)  # d = 1, V = 2r = 1.0
    assert np.isclose(float(pot.virial_pressure(s)[0]), expected)
    assert float(pot.virial_pressure(s)[0]) > 0.0  # r < sigma -> compression -> p > 0


# ---------------------------------------------------------------------------
# Mechanical relaxation.
# ---------------------------------------------------------------------------


def test_relaxation_separates_overlapping_pair():
    pot = Morse()
    s = _two_cells(dist=0.5)  # overlapping (sigma = 1.0)
    relax = MechanicalRelaxation(pot, max_steps=500, f_tol=1e-6)
    s2 = relax(s, dt=1.0, key=None)
    d0 = float(jnp.linalg.norm(s.position[0] - s.position[1]))
    d1 = float(jnp.linalg.norm(s2.position[0] - s2.position[1]))
    assert d1 > d0 and abs(d1 - 1.0) < 0.02  # relaxes to the contact equilibrium sigma=1.0


def test_relaxation_potential_none_is_a_no_op():
    # potential=None -> NoForce: every configuration is already an equilibrium, so an overlapping pair
    # (which Morse would separate) passes through unchanged.
    relax = MechanicalRelaxation(None, max_steps=500, f_tol=1e-6)
    assert isinstance(relax.potential, NoForce)
    s = _two_cells(dist=0.5)  # overlapping (sigma = 1.0)
    s2 = relax(s, dt=1.0, key=None)
    assert np.allclose(np.asarray(s2.position), np.asarray(s.position))


def test_relaxation_gradient_is_equilibrium_sensitivity():
    # The two-cell equilibrium separation is sigma = 2*radius, so d(separation)/d(radius) = 2.
    # Implicit diff must recover this regardless of the (fixed) start and solver path.
    relax = MechanicalRelaxation(Morse(), max_steps=800, f_tol=1e-8)

    def separation(radius):
        s = _two_cells(dist=1.0, radius=radius)
        s2 = relax(s, dt=1.0, key=None)
        return jnp.linalg.norm(s2.position[0] - s2.position[1])

    g = jax.grad(separation)(0.5)
    assert np.isclose(float(g), 2.0, atol=0.05)


def test_relaxation_adjoint_projects_out_translation_and_rotation():
    # The adjoint's rigid-body projection annihilates a global translation and a global rotation of
    # the alive cells (gauge / zero modes -> no gradient) while preserving a physical deformation.
    # Checked directly on the projector, in 2D and in 3D (a non-collinear triangle exercises all
    # three rotation generators and the Gram-Schmidt orthonormalization).
    from jax_morph.physics.mechanics.relaxation import _project, _rigid_body_modes

    def split(x, alive):
        basis = _rigid_body_modes(x, alive)
        a = alive.astype(x.dtype)[:, None]
        centered = (x - jnp.sum(x * a, axis=0, keepdims=True) / jnp.sum(a)) * a
        trans = a * jnp.ones_like(x)  # a global translation of the alive cells
        rot = jnp.zeros_like(x).at[:, 0].set(-centered[:, 1]).at[:, 1].set(centered[:, 0])
        assert np.allclose(np.asarray(_project(trans, alive, basis)), 0.0, atol=1e-6)
        assert np.allclose(np.asarray(_project(rot, alive, basis)), 0.0, atol=1e-6)
        assert not np.allclose(
            np.asarray(_project(centered, alive, basis)), 0.0
        )  # scaling survives

    split(_two_cells(dist=1.0).position, _two_cells(dist=1.0).alive)  # 2D
    s3 = build_state_from_model(_M()).init_empty(capacity=3, n_space_dim=3, n_types=1)
    s3 = s3.update(
        alive=s3.alive.at[:3].set(True),
        radius=s3.radius.at[:3].set(0.5),
        position=jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        celltype=s3.celltype.at[:3, 0].set(1.0),
    )
    split(s3.position, s3.alive)  # 3D


def test_relax_equilibrium_function_matches_the_step():
    # The free function relax_equilibrium is what the step wraps; called directly it returns the same
    # equilibrium positions (the step just writes them into the state).
    pot = Morse()
    s = _two_cells(dist=0.5)
    x = relax_equilibrium(pot, s, max_steps=500, f_tol=1e-6)
    d = float(jnp.linalg.norm(x[0] - x[1]))
    assert abs(d - 1.0) < 0.02


def test_relax_equilibrium_function_potential_none_is_a_no_op():
    # potential=None -> NoForce in the free function too (matching the step): an overlapping pair
    # passes through unchanged.
    s = _two_cells(dist=0.5)  # overlapping (sigma = 1.0)
    x = relax_equilibrium(None, s, max_steps=500, f_tol=1e-6)
    assert np.allclose(np.asarray(x), np.asarray(s.position))


def test_relaxation_reaches_force_tolerance():
    pot = Morse()
    relax = MechanicalRelaxation(pot, max_steps=800, f_tol=1e-6)
    s2 = relax(_two_cells(dist=0.7), dt=1.0, key=None)
    assert float(jnp.max(jnp.abs(pot.forces(s2)))) < 1e-4  # genuinely at |grad U| ~ 0


def test_relaxation_warns_on_non_convergence():
    # Hitting max_steps still above f_tol returns a non-equilibrium configuration (with a possibly
    # inaccurate implicit-diff gradient); a runtime RuntimeWarning must surface, emitted via
    # jax.debug.callback so it fires even from inside the jitted step, not just at trace time.
    relax = MechanicalRelaxation(Morse(), max_steps=1, f_tol=1e-12)  # cannot converge
    with pytest.warns(RuntimeWarning, match='did not reach equilibrium'):
        s2 = relax(_two_cells(dist=0.5), dt=1.0, key=None)
        _ = np.asarray(s2.position)  # materialize so the host callback runs


def test_relaxation_does_not_warn_when_converged():
    # The converged (common) path must stay warning-free: lax.cond keeps the callback off it.
    relax = MechanicalRelaxation(Morse(), max_steps=500, f_tol=1e-6)
    with warnings.catch_warnings():
        warnings.simplefilter('error')  # any warning here fails the test
        s2 = relax(_two_cells(dist=0.5), dt=1.0, key=None)
        _ = np.asarray(s2.position)


def test_relaxation_is_a_valid_quasistatic_model_step(key):
    # As a quasistatic step in a Model it advances one macro-step to the relaxed positions. The step
    # ignores the key, but Model.__call__ splits one, so pass a placeholder (simulate would inject it).
    pot = Morse()
    model = jxm.Model([MechanicalRelaxation(pot, max_steps=500, f_tol=1e-6)])
    s = _two_cells(dist=0.5)
    s1 = model(s, dt=1.0, key=key)
    d1 = float(jnp.linalg.norm(s1.position[0] - s1.position[1]))
    assert abs(d1 - 1.0) < 0.02 and np.isclose(float(s1.t), 1.0)


# ---------------------------------------------------------------------------
# Brownian dynamics (a reparameterized dynamic stochastic step).
# ---------------------------------------------------------------------------


def _brownian_state(bd, dist=1.2):
    s = build_state_from_model(bd).init_empty(capacity=2, n_space_dim=2, n_types=1)
    return s.update(
        alive=s.alive.at[:2].set(True),
        radius=s.radius.at[:2].set(0.5),
        position=jnp.array([[0.0, 0.0], [dist, 0.0]]),
    )


def test_brownian_potential_none_is_no_force(key):
    # potential=None -> NoForce: a free Brownian gas. Identical to passing NoForce explicitly, and
    # with kT=0 there is no drift either, so an overlapping pair (which Morse would push apart) stays.
    bd = BrownianDynamics(None, n_space_dim=2, kT=0.05)
    assert isinstance(bd.potential, NoForce)
    s = _brownian_state(bd, dist=0.6)  # overlapping (sigma = 1.0)
    default = jxm.Model([bd])(s, dt=0.1, key=key)
    explicit = jxm.Model([BrownianDynamics(NoForce(), n_space_dim=2, kT=0.05)])(s, dt=0.1, key=key)
    assert np.allclose(np.asarray(default.position), np.asarray(explicit.position))
    cold = jxm.Model([BrownianDynamics(None, n_space_dim=2, kT=0.0)])(s, dt=0.1, key=key)
    assert np.allclose(np.asarray(cold.position), np.asarray(s.position))  # no drift, no noise


def test_brownian_delta_shape_and_reproducible(key):
    pot = Morse()
    bd = BrownianDynamics(pot, n_space_dim=2, gamma=1.0, kT=0.05)
    s = _brownian_state(bd)
    d1 = bd(s, dt=0.01, key=key)  # a dynamic step returns a sparse delta state
    d2 = bd(s, dt=0.01, key=key)
    d_other = bd(s, dt=0.01, key=jax.random.PRNGKey(1))
    assert d1.position.shape == (2, 2) and d1.brownian_dx.shape == (2, 2)
    assert np.allclose(np.asarray(d1.position), np.asarray(d2.position))  # same key -> same draw
    assert not np.allclose(
        np.asarray(d1.position), np.asarray(d_other.position)
    )  # key drives noise


def test_brownian_trace_round_trips(key):
    # The recorded trace (noise + realized dx) must survive replay -> trace_from_state, or logp would
    # silently score reset defaults. check_stochastic_step drives one step and asserts the round-trip.
    bd = BrownianDynamics(Morse(), n_space_dim=2, gamma=1.0, kT=0.05)
    s = _brownian_state(bd)
    recorded = check_stochastic_step(bd, s, dt=0.01, key=key)
    assert set(recorded) == {'brownian_xi', 'brownian_dx'}


def test_brownian_n_space_dim_must_match_state(key):
    # n_space_dim sizes the recorded xi/dx trace fields at build time; a mismatch with the state's
    # actual spatial dimension is caught early with a clear error, not an obscure broadcast failure.
    bd = BrownianDynamics(Morse(), n_space_dim=3)  # built for 3D
    s = _brownian_state(bd)  # a 2D state
    with pytest.raises(ValueError, match='n_space_dim'):
        bd(s, dt=0.01, key=key)


def test_check_stochastic_step_catches_discrete_replay_forgetting_a_derived_field(key):
    # The two-run differential in check_stochastic_step catches a DISCRETE replay that records the
    # sampled action but forgets a derived trace field - a single reset-run inspection could not,
    # since the forgotten field would just read its default and look co-emitted.
    class ForgetfulDivide(jxm.StochasticStep):
        step_type = jxm.StepType.DISCRETE
        p: jax.Array

        def trace_writes(self):
            return (
                jxm.StateFieldSpec('fd_divided', shape=(), heritable=False),
                jxm.StateFieldSpec(
                    'fd_aux', shape=(), heritable=False
                ),  # derived, never co-emitted
            )

        def _dist(self, state, dt):
            return self.p * jnp.ones_like(state.radius)

        def sample_trace(self, state, *, dt, key):
            divided = jxm.ad_utils.sample_bernoulli_st(key, self._dist(state, dt))
            return {'fd_divided': divided * state.alive.astype(divided.dtype)}

        def replay(self, state, trace, *, dt, pathwise):
            return state.update(fd_divided=trace['fd_divided'])  # forgets to co-emit fd_aux

        def logp(self, state, trace, dt):
            return jnp.sum(jxm.ad_utils.bernoulli_logp(trace['fd_divided'], self._dist(state, dt)))

    step = ForgetfulDivide(p=jnp.array(0.5))
    s = build_state_from_model(jxm.Model([step])).init_empty(capacity=3, n_space_dim=2, n_types=1)
    s = s.update(alive=s.alive.at[:2].set(True), radius=s.radius.at[:2].set(0.5))
    with pytest.raises(AssertionError, match='not co-emitted'):
        check_stochastic_step(step, s, key=key)


def test_brownian_tag_namespaces_trace_fields(key):
    # Two Brownian steps coexist when their trace fields are namespaced by distinct tags.
    a = BrownianDynamics(Morse(), n_space_dim=2, tag='a')
    b = BrownianDynamics(Morse(), n_space_dim=2, tag='b')
    model = jxm.Model([a, b])  # constructs iff the trace fields do not collide
    s = build_state_from_model(model).init_empty(capacity=2, n_space_dim=2, n_types=1)
    s = s.update(alive=s.alive.at[:2].set(True), radius=s.radius.at[:2].set(0.5))
    s1 = model(s, dt=0.01, key=key)
    assert hasattr(s1, 'a_dx') and hasattr(s1, 'b_dx')


def test_brownian_logp_is_gaussian_over_the_recorded_displacement(key):
    # Beyond the cutoff the drift is zero (mean = 0), so logp reduces to the standard-normal-scaled
    # Gaussian of the recorded dx with std = sqrt(2 kT dt / gamma), summed over alive cells and dims.
    dt, gamma, kT = 0.01, 1.0, 0.05
    bd = BrownianDynamics(Morse(), n_space_dim=2, gamma=gamma, kT=kT)
    model = jxm.Model([bd])
    s0 = _brownian_state(bd, dist=3.0)  # far apart -> zero force -> zero mean
    s1 = model(s0, dt=dt, key=key)

    dx = np.asarray(s1.brownian_dx)
    std = np.sqrt(2.0 * kT * dt / gamma)
    # both cells alive, all dims
    expect = float(np.sum(-0.5 * (dx / std) ** 2 - np.log(std) - 0.5 * np.log(2 * np.pi)))
    assert np.isclose(float(jxm.transition_logp(model, s0, s1, dt, score='all')), expect)


def test_brownian_logp_includes_the_nonzero_drift(key):
    # Within the well the drift mean = dt * forces / gamma is nonzero, so logp must score dx under
    # N(mean, std**2), not N(0, std**2). Check against the reference built from that exact drift - a
    # wrong sign or coefficient in the mean would slip past the zero-drift test above.
    dt, gamma, kT = 0.01, 1.0, 0.05
    pot = Morse()
    bd = BrownianDynamics(pot, n_space_dim=2, gamma=gamma, kT=kT)
    model = jxm.Model([bd])
    s0 = _brownian_state(bd, dist=1.2)  # within the well -> nonzero force -> nonzero mean
    s1 = model(s0, dt=dt, key=key)

    mean = np.asarray(dt * pot.forces(s0) / gamma)
    std = np.sqrt(2.0 * kT * dt / gamma)
    dx = np.asarray(s1.brownian_dx)
    assert not np.allclose(mean, 0.0)  # the drift is genuinely nonzero in this configuration
    expect = float(np.sum(-0.5 * ((dx - mean) / std) ** 2 - np.log(std) - 0.5 * np.log(2 * np.pi)))
    assert np.isclose(float(jxm.transition_logp(model, s0, s1, dt, score='all')), expect)


def test_brownian_logp_is_finite_at_zero_temperature(key):
    # kT=0 (deterministic overdamped relaxation, no noise): logp is finite - the deterministic
    # displacement contributes 0, not a NaN from a zero-variance Gaussian kernel.
    bd = BrownianDynamics(Morse(), n_space_dim=2, kT=0.0)
    s0 = _brownian_state(bd, dist=1.2)
    model = jxm.Model([bd])
    s1 = model(s0, dt=0.5, key=key)
    assert np.isfinite(float(jxm.transition_logp(model, s0, s1, dt=0.5, score='all')))


def test_brownian_logp_gradient_reaches_kT(key):
    # The score-function gradient of a Brownian transition reaches the noise-scale kT; check it
    # against a finite difference of the same (fixed recorded action) log-density.
    dt = 0.01
    pot = Morse()
    s0 = _brownian_state(BrownianDynamics(pot, n_space_dim=2), dist=1.2)
    s1 = jxm.Model([BrownianDynamics(pot, n_space_dim=2, kT=0.05)])(s0, dt=dt, key=key)

    def logp_of_kT(kt):
        m = jxm.Model([BrownianDynamics(pot, n_space_dim=2, gamma=1.0, kT=kt)])
        return jxm.transition_logp(m, s0, s1, dt, score='all')

    g = float(jax.grad(lambda kt: logp_of_kT(kt))(0.05))
    fd = _central_diff(lambda kt: float(logp_of_kT(kt)), 0.05, eps=1e-6)
    assert g != 0.0 and np.isclose(g, fd, atol=1e-4)


def test_brownian_logp_gradient_reaches_potential_and_gamma(key):
    # The drift is -grad U / gamma, so the transition's logp is differentiable w.r.t. the potential's
    # parameters (a traced epsilon) and gamma; both gradients are finite and nonzero.
    dt = 0.01
    bd = BrownianDynamics(Morse(epsilon=jnp.array(3.0)), n_space_dim=2, gamma=jnp.array(1.0))
    model = jxm.Model([bd])
    s0 = _brownian_state(bd, dist=1.2)  # within the well -> nonzero force -> mean depends on params
    s1 = model(s0, dt=dt, key=key)

    g = eqx.filter_grad(lambda m: jxm.transition_logp(m, s0, s1, dt, score='all'))(model)
    assert (
        np.isfinite(float(g.steps[0].potential.epsilon))
        and float(g.steps[0].potential.epsilon) != 0.0
    )
    assert np.isfinite(float(g.steps[0].gamma)) and float(g.steps[0].gamma) != 0.0


def test_brownian_simulate_is_pathwise_differentiable(key):
    # The forward rollout stays pathwise-differentiable through the reparameterized Brownian step:
    # with the noise frozen, the displacement scales with the noise amplitude, so d(objective)/d(kT)
    # is a live, finite, nonzero pathwise gradient (no score-function term involved).
    bd = BrownianDynamics(Morse(), n_space_dim=2, gamma=1.0, kT=jnp.array(0.05))
    s0 = _brownian_state(bd, dist=3.0)  # zero drift -> displacement is pure noise * amplitude

    def objective(kt):
        m = jxm.Model([BrownianDynamics(Morse(), n_space_dim=2, gamma=1.0, kT=kt)])
        final = jxm.simulate(m, s0, n_steps=5, dt=0.01, key=key)
        alive = s0.alive.astype(final.position.dtype)[:, None]
        return jnp.sum((final.position**2) * alive)

    g = float(jax.grad(objective)(0.05))
    assert np.isfinite(g) and g != 0.0


# ---------------------------------------------------------------------------
# gaussian_logp density helper (ships with the Brownian step).
# ---------------------------------------------------------------------------


def test_gaussian_logp_matches_reference_and_gradient():
    x = jnp.array([0.3, -1.2, 2.0])
    mean = jnp.array([0.0, -1.0, 1.5])
    std = 0.7
    ref = -0.5 * ((x - mean) / std) ** 2 - np.log(std) - 0.5 * np.log(2 * np.pi)
    assert np.allclose(np.asarray(jxm.ad_utils.gaussian_logp(x, mean, std)), np.asarray(ref))
    # d/dmean log N = (x - mean) / std^2
    g = jax.grad(lambda m: jxm.ad_utils.gaussian_logp(x, m, std).sum())(mean)
    assert np.allclose(np.asarray(g), np.asarray((x - mean) / std**2))
