"""Acceptance tests for conditional transition trace scoring.

A stochastic step is a ``StochasticStep``: it records a **trace** (the exogenous noise plus the
realized action) into ephemeral trace fields and owns a single ``replay`` that is the one source of
its effect. ``transition_logp(model, state, next_state, dt, *, score=None)`` re-runs the macro-step
on the detached conditioning ``state`` with parameters live, replays each stochastic step from its
recorded trace, and sums the selected steps' ``logp``. These toy steps exercise the contract: score-function
gradients, the wide MLE gradient through a deterministic upstream, the within-step pathwise/score
hybrid, a downstream reading an upstream trace, phase-correct dynamic scoring, trace reset over a
rollout, importance sampling, and step selection.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jax_morph as jxm

# A shared spec so a producer's write and a consumer's read merge to the same field.
EMIT = jxm.StateFieldSpec('emit', shape=(), heritable=False)
# A shared physical field a reparameterized step kicks and a downstream step reads.
U = jxm.StateFieldSpec('u', shape=(), heritable=False)


def _gaussian_logp(x, mean, std):
    """Elementwise Gaussian log-density (test-local; the core ships this with the Brownian step)."""
    z = (x - mean) / std
    return -0.5 * z * z - jnp.log(std) - 0.5 * jnp.log(2.0 * jnp.pi)


# ---------------------------------------------------------------------------
# Toy stochastic steps built on the trace contract.
# ---------------------------------------------------------------------------


class MaybeDivide(jxm.StochasticStep):
    """A minimal discrete stochastic step: each live cell divides with probability ``p``.

    The trace is ``{divided, divide_eligible}`` - the 0/1 action and the eligible-to-divide mask
    (alive at decision time) - both ephemeral trace fields written by ``replay``. ``_dist`` is the
    single source of ``p``; ``logp`` scores the recorded action under it, over the eligible cells.
    """

    step_type = jxm.StepType.DISCRETE
    p: jax.Array

    def trace_writes(self):
        return (
            jxm.StateFieldSpec('divided', shape=(), heritable=False),
            jxm.StateFieldSpec('divide_eligible', shape=(), heritable=False),
        )

    def _dist(self, state, dt):
        return self.p * jnp.ones_like(state.radius)

    def sample_trace(self, state, *, dt, key):
        prob = self._dist(state, dt)
        divided = jxm.ad_utils.sample_bernoulli_st(key, prob)
        alive_f = state.alive.astype(divided.dtype)
        return {'divided': divided * alive_f, 'divide_eligible': alive_f}

    def replay(self, state, trace, *, dt, pathwise):
        return state.update(divided=trace['divided'], divide_eligible=trace['divide_eligible'])

    def logp(self, state, trace, dt):
        prob = self._dist(state, dt)
        return jnp.sum(
            jxm.ad_utils.bernoulli_logp(trace['divided'], prob) * trace['divide_eligible']
        )


class SetProb(jxm.SimulationStep):
    """Deterministic quasistatic step: set a per-cell probability field ``prob`` from a param ``w``.

    Stands in for an upstream deterministic decision chain; the stochastic step downstream reads
    ``prob``, so the re-run must recompute it through ``w`` for the gradient to reach ``w``.
    """

    step_type = jxm.StepType.QUASISTATIC
    w: jax.Array

    def state_writes(self):
        return (jxm.StateFieldSpec('prob', shape=()),)

    def __call__(self, state, *, dt, key):
        return state.set('prob', jax.nn.sigmoid(self.w) * jnp.ones_like(state.radius))


class Flip(jxm.StochasticStep):
    """Discrete stochastic step reading an upstream ``prob`` field (no params of its own)."""

    step_type = jxm.StepType.DISCRETE

    def state_reads(self):
        return (jxm.StateFieldSpec('prob', shape=()),)

    def trace_writes(self):
        return (
            jxm.StateFieldSpec('flipped', shape=(), heritable=False),
            jxm.StateFieldSpec('flip_eligible', shape=(), heritable=False),
        )

    def _dist(self, state, dt):
        return state.prob

    def sample_trace(self, state, *, dt, key):
        flipped = jxm.ad_utils.sample_bernoulli_st(key, self._dist(state, dt))
        alive_f = state.alive.astype(flipped.dtype)
        return {'flipped': flipped * alive_f, 'flip_eligible': alive_f}

    def replay(self, state, trace, *, dt, pathwise):
        return state.update(flipped=trace['flipped'], flip_eligible=trace['flip_eligible'])

    def logp(self, state, trace, dt):
        return jnp.sum(
            jxm.ad_utils.bernoulli_logp(trace['flipped'], self._dist(state, dt))
            * trace['flip_eligible']
        )


class Emit(jxm.StochasticStep):
    """Upstream discrete stochastic step: sample a bit and write it to a field read downstream."""

    step_type = jxm.StepType.DISCRETE
    q: jax.Array

    def trace_writes(self):
        return (EMIT, jxm.StateFieldSpec('emit_eligible', shape=(), heritable=False))

    def _dist(self, state, dt):
        return jax.nn.sigmoid(self.q) * jnp.ones_like(state.radius)

    def sample_trace(self, state, *, dt, key):
        bit = jxm.ad_utils.sample_bernoulli_st(key, self._dist(state, dt))
        alive_f = state.alive.astype(bit.dtype)
        return {'emit': bit * alive_f, 'emit_eligible': alive_f}

    def replay(self, state, trace, *, dt, pathwise):
        return state.update(emit=trace['emit'], emit_eligible=trace['emit_eligible'])

    def logp(self, state, trace, dt):
        return jnp.sum(
            jxm.ad_utils.bernoulli_logp(trace['emit'], self._dist(state, dt))
            * trace['emit_eligible']
        )


class React(jxm.StochasticStep):
    """Downstream discrete stochastic step whose per-cell probability depends on ``emit``.

    So the re-run must reconstruct ``emit`` into React's conditioning state: Emit's ``replay`` writes
    the (frozen) recorded ``emit`` before React scores, and React reads it back.
    """

    step_type = jxm.StepType.DISCRETE
    w: jax.Array

    def state_reads(self):
        return (EMIT,)

    def trace_writes(self):
        return (
            jxm.StateFieldSpec('react', shape=(), heritable=False),
            jxm.StateFieldSpec('react_eligible', shape=(), heritable=False),
        )

    def _dist(self, state, dt):
        return jax.nn.sigmoid(self.w + 5.0 * state.emit)

    def sample_trace(self, state, *, dt, key):
        bit = jxm.ad_utils.sample_bernoulli_st(key, self._dist(state, dt))
        alive_f = state.alive.astype(bit.dtype)
        return {'react': bit * alive_f, 'react_eligible': alive_f}

    def replay(self, state, trace, *, dt, pathwise):
        return state.update(react=trace['react'], react_eligible=trace['react_eligible'])

    def logp(self, state, trace, dt):
        return jnp.sum(
            jxm.ad_utils.bernoulli_logp(trace['react'], self._dist(state, dt))
            * trace['react_eligible']
        )


class Kick(jxm.StochasticStep):
    """A reparameterized dynamic step: kick the scalar field ``u`` by ``dx = mean + std * xi``.

    ``std = exp(log_std)`` and ``mean = mean_coeff * u`` (both from ``_dist``). The recorded trace
    is the exogenous noise ``xi`` and the realized displacement ``dx`` (namespaced by ``tag`` so
    several instances coexist), both additive dynamic trace fields defaulting to ``0``.
    ``sample_trace`` draws only the noise; ``replay`` derives ``dx`` and records both. With
    ``pathwise=True`` it recomputes ``dx`` from ``xi`` with the live ``log_std``; with
    ``pathwise=False`` it reuses the frozen recorded ``dx``.
    """

    step_type = jxm.StepType.DYNAMIC
    log_std: jax.Array
    tag: str = eqx.field(static=True, default='kick')
    mean_coeff: float = eqx.field(static=True, default=0.0)

    @property
    def _xi(self):
        return f'{self.tag}_xi'

    @property
    def _dx(self):
        return f'{self.tag}_dx'

    def state_writes(self):
        return (U,)

    def trace_writes(self):
        return (
            jxm.StateFieldSpec(self._xi, shape=(), heritable=False),
            jxm.StateFieldSpec(self._dx, shape=(), heritable=False),
        )

    def _dist(self, state, dt):
        mean = self.mean_coeff * state.u
        std = jnp.exp(self.log_std) * jnp.ones_like(state.radius)
        return mean, std

    def sample_trace(self, state, *, dt, key):
        return {self._xi: jax.random.normal(key, state.radius.shape)}

    def replay(self, state, trace, *, dt, pathwise):
        if pathwise:
            mean, std = self._dist(state, dt)
            dx = mean + std * trace[self._xi]  # recompute from the noise with live params
        else:
            dx = trace[self._dx]  # frozen realized displacement
        return state.deltas(**{'u': dx, self._xi: trace[self._xi], self._dx: dx})

    def logp(self, state, trace, dt):
        mean, std = self._dist(state, dt)
        alive_f = state.alive.astype(std.dtype)
        return jnp.sum(_gaussian_logp(trace[self._dx], mean, std) * alive_f)


class DivU(jxm.StochasticStep):
    """Discrete stochastic step whose division probability reads the kicked field ``u``.

    ``prob = sigmoid(b + c * u)`` - so a reparameterized upstream ``Kick`` (which sets ``u``) reaches
    this step's ``logp`` pathwise when it is replayed live.
    """

    step_type = jxm.StepType.DISCRETE
    b: jax.Array
    c: float = eqx.field(static=True, default=1.0)

    def state_reads(self):
        return (U,)

    def trace_writes(self):
        return (
            jxm.StateFieldSpec('divu', shape=(), heritable=False),
            jxm.StateFieldSpec('divu_eligible', shape=(), heritable=False),
        )

    def _dist(self, state, dt):
        return jax.nn.sigmoid(self.b + self.c * state.u)

    def sample_trace(self, state, *, dt, key):
        out = jxm.ad_utils.sample_bernoulli_st(key, self._dist(state, dt))
        alive_f = state.alive.astype(out.dtype)
        return {'divu': out * alive_f, 'divu_eligible': alive_f}

    def replay(self, state, trace, *, dt, pathwise):
        return state.update(divu=trace['divu'], divu_eligible=trace['divu_eligible'])

    def logp(self, state, trace, dt):
        return jnp.sum(
            jxm.ad_utils.bernoulli_logp(trace['divu'], self._dist(state, dt))
            * trace['divu_eligible']
        )


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _seed(model, capacity, n_live, *, n_space_dim=2, **fields):
    state = jxm.build_state_from_model(model).init_empty(
        capacity=capacity, n_space_dim=n_space_dim, n_types=1
    )
    state = state.set('alive', state.alive.at[:n_live].set(True))
    if fields:
        state = state.update(**{k: v for k, v in fields.items()})
    return state


# ---------------------------------------------------------------------------
# Acceptance tests.
# ---------------------------------------------------------------------------


def test_score_function_gradient_reaches_policy_param(key):
    # A single scored discrete step: logp scores the RECORDED action (not a re-sample) and its
    # gradient equals the analytic Bernoulli score sum_deciders (outcome/p - (1-outcome)/(1-p)).
    p0 = 0.3
    model = jxm.Model([MaybeDivide(p=jnp.array(p0))])
    s0 = _seed(model, capacity=4, n_live=3)

    s1 = model(s0, dt=1.0, key=key)  # forward samples; the action + eligibility mask ride in s1
    assert set(np.asarray(s1.divided).tolist()) <= {0.0, 1.0}
    assert np.allclose(np.asarray(s1.divide_eligible), np.asarray(s0.alive).astype(float))

    alive = np.asarray(s0.alive)
    outcome = np.asarray(s1.divided)
    expect = float(np.sum(alive * (outcome * np.log(p0) + (1 - outcome) * np.log(1 - p0))))
    assert np.isclose(float(jxm.transition_logp(model, s0, s1, 1.0)), expect)

    g = eqx.filter_grad(lambda m: jxm.transition_logp(m, s0, s1, 1.0))(model)
    analytic = float(np.sum(alive * (outcome / p0 - (1 - outcome) / (1 - p0))))
    assert np.isclose(float(g.steps[0].p), analytic)


def test_wide_gradient_reaches_deterministic_upstream_param(key):
    # MLE / wide gradient: prob is set by a deterministic quasistatic step from w and only READ by
    # the stochastic step. A scorer reading the recorded (detached) prob would give dlogp/dw = 0; the
    # live re-run recomputes prob through w, so the gradient reaches w. Analytic: sum(outcome - p).
    w0 = 0.4
    prob = float(1.0 / (1.0 + np.exp(-w0)))
    model = jxm.Model([SetProb(w=jnp.array(w0)), Flip()])
    s0 = _seed(model, capacity=4, n_live=3)

    s1 = model(s0, dt=1.0, key=key)
    assert np.allclose(np.asarray(s1.prob[:3]), prob)  # prob really came from w

    g = eqx.filter_grad(lambda m: jxm.transition_logp(m, s0, s1, 1.0))(model)
    alive = np.asarray(s0.alive)
    outcome = np.asarray(s1.flipped)
    analytic = float(np.sum(alive * (outcome - prob)))
    assert np.isclose(float(g.steps[0].w), analytic)
    assert g.steps[0].w != 0.0  # the re-run is what makes this nonzero


def test_within_step_hybrid_pathwise_through_unscored_reparam_into_scored_discrete(key):
    # Pipeline: Kick (reparameterized, dynamic, score_by_default=False) -> DivU (discrete, scored).
    # Kick sets u = std*xi; DivU's prob = sigmoid(b + c*u). Scoring only DivU, the reparameterized
    # Kick is replayed pathwise, so the diffusivity param log_std reaches DivU.logp. Analytic:
    # d logp / d log_std = sum_deciders (divu - prob) * c * dx (since d(logit)/d log_std = c*dx).
    log_std0, b0, c0 = np.log(0.7), 0.2, 1.5
    kick = Kick(log_std=jnp.array(log_std0), mean_coeff=0.0, score_by_default=False)
    div = DivU(b=jnp.array(b0), c=c0, score_by_default=True)
    model = jxm.Model([kick, div])
    s0 = _seed(model, capacity=5, n_live=4)

    s1 = model(s0, dt=1.0, key=key)
    alive = np.asarray(s0.alive)
    dx = np.asarray(s1.kick_dx)
    u = np.asarray(s1.u)
    assert np.allclose(u, dx)  # u started at 0, so the recorded displacement is u
    divu = np.asarray(s1.divu)
    prob = 1.0 / (1.0 + np.exp(-(b0 + c0 * u)))

    # (a) score only DivU: the gradient wrt log_std flows pathwise through the unscored Kick.
    g_a = eqx.filter_grad(lambda m: jxm.transition_logp(m, s0, s1, 1.0, score=None))(model)
    analytic = float(np.sum(alive * (divu - prob) * c0 * dx))
    assert np.isclose(float(g_a.steps[0].log_std), analytic)
    assert g_a.steps[0].log_std != 0.0

    # (b) when Kick is instead scored (frozen), DivU's own term contributes ZERO log_std-gradient:
    # scoring {Kick, DivU} gives the same log_std-gradient as scoring {Kick} alone.
    g_both = eqx.filter_grad(lambda m: jxm.transition_logp(m, s0, s1, 1.0, score=[0, 1]))(model)
    g_kick = eqx.filter_grad(lambda m: jxm.transition_logp(m, s0, s1, 1.0, score=[0]))(model)
    assert np.isclose(float(g_both.steps[0].log_std), float(g_kick.steps[0].log_std))
    assert g_kick.steps[0].log_std != 0.0  # Kick's own Gaussian logp credits log_std


def test_downstream_reads_upstream_trace_no_double_count(key):
    # Emit -> React, both scored: React's prob depends on Emit's recorded bit. The re-run replays
    # Emit's frozen bit into React's conditioning, so React's logp credits Emit's param q only
    # through Emit's own term (no straight-through double-count).
    q0, w0 = 0.0, 0.3
    model = jxm.Model([Emit(q=jnp.array(q0)), React(w=jnp.array(w0))])
    s0 = _seed(model, capacity=8, n_live=6)

    s1 = model(s0, dt=1.0, key=key)
    emit = np.asarray(s1.emit)
    react = np.asarray(s1.react)
    alive = np.asarray(s0.alive)
    assert set(emit.tolist()) <= {0.0, 1.0} and set(react.tolist()) <= {0.0, 1.0}
    assert {0.0, 1.0} <= set(emit[:6].tolist())  # a mix of bits, so React's prob really varies

    g = eqx.filter_grad(lambda m: jxm.transition_logp(m, s0, s1, 1.0, score='all'))(model)

    p2 = 1.0 / (1.0 + np.exp(-(w0 + 5.0 * emit)))  # React uses the RECORDED emit bit
    dw = float(np.sum(alive * (react - p2)))
    assert np.isclose(float(g.steps[1].w), dw)

    p1 = 1.0 / (1.0 + np.exp(-q0))  # Emit's prob; React's logp does not depend on q
    dq = float(np.sum(alive * (emit - p1)))
    assert np.isclose(float(g.steps[0].q), dq)


def test_phase_correct_dynamic_scores_at_shared_post_quasistatic_state(key):
    # Two dynamic Kicks with mean = 0.5 * u both condition on the SAME post-quasistatic u (seeded
    # 1.0), not a sequential sweep - the second must NOT see the first's delta. If it did, its mean
    # would use u = 1 + dx1 and its logp would differ. Both write the shared u (dynamic accumulate).
    u0 = 1.0
    ka = Kick(log_std=jnp.array(np.log(0.5)), tag='a', mean_coeff=0.5, score_by_default=True)
    kb = Kick(log_std=jnp.array(np.log(0.9)), tag='b', mean_coeff=0.5, score_by_default=True)
    model = jxm.Model([ka, kb])
    s0 = _seed(model, capacity=4, n_live=3)
    s0 = s0.set('u', jnp.full_like(s0.u, u0))

    s1 = model(s0, dt=1.0, key=key)
    alive = np.asarray(s0.alive)
    dxa, dxb = np.asarray(s1.a_dx), np.asarray(s1.b_dx)
    assert np.allclose(np.asarray(s1.u), u0 + dxa + dxb)  # both deltas accumulate once

    # logp computed at the shared u0 for BOTH steps (mean = 0.5 * u0), not sequentially.
    mean = 0.5 * u0
    exp_a = float(
        np.sum(alive * (-0.5 * ((dxa - mean) / 0.5) ** 2 - np.log(0.5) - 0.5 * np.log(2 * np.pi)))
    )
    exp_b = float(
        np.sum(alive * (-0.5 * ((dxb - mean) / 0.9) ** 2 - np.log(0.9) - 0.5 * np.log(2 * np.pi)))
    )
    assert np.isclose(float(jxm.transition_logp(model, s0, s1, 1.0, score='all')), exp_a + exp_b)


def test_trace_reset_records_per_step_value_not_running_sum(key):
    # Over a multi-step rollout the recorded additive trace field holds EACH step's displacement, not
    # a running sum - because the model resets trace fields at the start of every macro-step.
    kick = Kick(log_std=jnp.array(np.log(0.3)), mean_coeff=0.0, score_by_default=True)
    model = jxm.Model([kick])
    s0 = _seed(model, capacity=3, n_live=2)

    traj = jxm.simulate(model, s0, n_steps=4, dt=1.0, key=key, history=True)
    dxs = np.asarray(traj.kick_dx[1:])  # one trace record per macro-step
    us = np.asarray(traj.u[1:])
    # u is the cumulative walk; kick_dx is the per-step increment => u_t == sum of dx_0..dx_t.
    assert np.allclose(us, np.cumsum(dxs, axis=0))
    # and no recorded dx equals the running sum (they differ once the walk moves).
    assert not np.allclose(dxs[1:], us[1:])


def test_pathwise_gradient_through_stochastic_simulate(key):
    # The other optimization primitive: the forward rollout stays pathwise-differentiable THROUGH a
    # reparameterized stochastic step. Kick sets u += exp(log_std) * xi with the noise xi frozen, so
    # d/d(log_std) flows through the recomputed value (no score-function term, just the pathwise one).
    log_std0 = np.log(0.6)
    model = jxm.Model([Kick(log_std=jnp.array(log_std0), mean_coeff=0.0)])
    s0 = _seed(model, capacity=4, n_live=3)

    def objective(m):
        final = jxm.simulate(m, s0, n_steps=5, dt=1.0, key=key)
        return jnp.sum(final.u * s0.alive.astype(final.u.dtype))

    # u_final = exp(log_std) * sum_t xi_t (mean 0), so summed over alive cells the objective scales as
    # exp(log_std); hence d(objective)/d(log_std) == the objective value itself (same frozen xi).
    j = float(objective(model))
    g = eqx.filter_grad(objective)(model)
    assert g.steps[0].log_std != 0.0  # the reparameterized forward carries a live pathwise gradient
    assert np.isclose(float(g.steps[0].log_std), j)


def test_importance_sampling_ratio_from_a_fixed_recorded_action(key):
    # Score a FIXED recorded discrete action at a shifted parameter: only the recomputed distribution
    # moves, so logp(p_new) - logp(p_old) is the well-defined importance-sampling log-ratio.
    p_old, p_new = 0.3, 0.55
    m_old = jxm.Model([MaybeDivide(p=jnp.array(p_old))])
    s0 = _seed(m_old, capacity=5, n_live=4)
    s1 = m_old(s0, dt=1.0, key=key)  # the recorded action is fixed in s1

    m_new = jxm.Model([MaybeDivide(p=jnp.array(p_new))])
    alive = np.asarray(s0.alive)
    outcome = np.asarray(s1.divided)

    lp_old = float(jxm.transition_logp(m_old, s0, s1, 1.0))
    lp_new = float(jxm.transition_logp(m_new, s0, s1, 1.0))
    ratio = lp_new - lp_old
    expect = float(
        np.sum(
            alive
            * (
                outcome * (np.log(p_new) - np.log(p_old))
                + (1 - outcome) * (np.log(1 - p_new) - np.log(1 - p_old))
            )
        )
    )
    assert np.isclose(ratio, expect)


def test_score_selection_none_all_explicit_and_two_same_class_by_index(key):
    # score=None scores the score_by_default set; 'all' scores every stochastic step; an explicit
    # index set scores exactly those; two same-class stochastic steps are addressed by index.
    da = MaybeDivide(p=jnp.array(0.3), score_by_default=True)
    db = MaybeDivide(p=jnp.array(0.6), score_by_default=False)
    # Two MaybeDivide instances would share trace field names; namespace via distinct classes here is
    # unnecessary - we instead use one MaybeDivide and one Flip (reads prob) to keep fields disjoint.
    model = jxm.Model([SetProb(w=jnp.array(0.4)), da, Flip()])
    s0 = _seed(model, capacity=5, n_live=4)
    s1 = model(s0, dt=1.0, key=key)

    idx_da, idx_flip = 1, 2
    lp_da = float(jxm.transition_logp(model, s0, s1, 1.0, score=[idx_da]))
    lp_flip = float(jxm.transition_logp(model, s0, s1, 1.0, score=[idx_flip]))
    lp_all = float(jxm.transition_logp(model, s0, s1, 1.0, score='all'))
    lp_default = float(jxm.transition_logp(model, s0, s1, 1.0, score=None))

    # da has score_by_default=True, Flip defaults to True => None == all here.
    assert np.isclose(lp_all, lp_da + lp_flip)
    assert np.isclose(lp_default, lp_all)
    # selecting a non-stochastic step (SetProb at index 0) is an error.
    with pytest.raises(ValueError, match='non-stochastic'):
        jxm.transition_logp(model, s0, s1, 1.0, score=[0])

    # score_by_default is honoured: with db (default False) unflagged, None skips it.
    model2 = jxm.Model([db, Flip()])  # db index 0 default False, Flip index 1 default True
    s0b = _seed(model2, capacity=5, n_live=4)
    s0b = s0b.set('prob', jnp.full_like(s0b.radius, 0.5))
    s1b = model2(s0b, dt=1.0, key=key)
    lp_none = float(jxm.transition_logp(model2, s0b, s1b, 1.0, score=None))
    lp_flip_only = float(jxm.transition_logp(model2, s0b, s1b, 1.0, score=[1]))
    assert np.isclose(lp_none, lp_flip_only)  # db skipped by default

    # two same-class stochastic steps (two Kicks, namespaced by tag) are addressed by index.
    kick_model = jxm.Model(
        [
            Kick(log_std=jnp.array(np.log(0.4)), tag='a', score_by_default=True),
            Kick(log_std=jnp.array(np.log(0.7)), tag='b', score_by_default=True),
        ]
    )
    sk = _seed(kick_model, capacity=4, n_live=3)
    sk1 = kick_model(sk, dt=1.0, key=key)
    lp_k0 = float(jxm.transition_logp(kick_model, sk, sk1, 1.0, score=[0]))
    lp_k1 = float(jxm.transition_logp(kick_model, sk, sk1, 1.0, score=[1]))
    assert np.isclose(
        float(jxm.transition_logp(kick_model, sk, sk1, 1.0, score='all')), lp_k0 + lp_k1
    )
    assert not np.isclose(lp_k0, lp_k1)  # distinct steps -> distinct scores, resolved by index


def test_score_selection_rejects_boolean_masks_and_out_of_range(key):
    # A boolean mask (Python, numpy or jax bools) is rejected with a clear error rather than
    # silently reinterpreted as integer indices (bool subclasses int, so [False, False, True] would
    # otherwise read as the index set {0, 1}); an out-of-range index is reported.
    model = jxm.Model([SetProb(w=jnp.array(0.4)), MaybeDivide(p=jnp.array(0.3)), Flip()])
    s0 = _seed(model, capacity=5, n_live=4)
    s1 = model(s0, dt=1.0, key=key)

    masks = ([False, False, True], np.array([False, False, True]), jnp.array([False, False, True]))
    for mask in masks:
        with pytest.raises(ValueError, match='boolean masks'):
            jxm.transition_logp(model, s0, s1, 1.0, score=mask)
    with pytest.raises(ValueError, match='out-of-range'):
        jxm.transition_logp(model, s0, s1, 1.0, score=[5])


def test_trace_field_collision_and_reserved_name_are_rejected_at_construction():
    # Two stochastic steps declaring the same trace field name would share one allocation and clobber
    # each other's recorded action -> rejected at construction (the tests namespace by tag to avoid).
    with pytest.raises(ValueError, match='unique per step'):
        jxm.Model([MaybeDivide(p=jnp.array(0.3)), MaybeDivide(p=jnp.array(0.6))])

    # a trace field shadowing a base (or physical) field would be wiped by the macro-step reset.
    class BadTrace(jxm.StochasticStep):
        step_type = jxm.StepType.DISCRETE

        def trace_writes(self):
            return (jxm.RADIUS,)  # 'radius' is a base field

    with pytest.raises(ValueError, match='collides'):
        jxm.Model([BadTrace()])


def test_dynamic_trace_field_with_nonzero_default_is_rejected_at_construction():
    # A dynamic (additive) trace field must default to 0: the per-macro-step reset then additive
    # accumulate would otherwise record default + value (and leave the raw default on dead slots),
    # silently corrupting exactly the quantity logp scores. Rejected at construction.
    class BadDynamicTrace(jxm.StochasticStep):
        step_type = jxm.StepType.DYNAMIC

        def state_writes(self):
            return (U,)

        def trace_writes(self):
            return (jxm.StateFieldSpec('bad_dx', shape=(), heritable=False, default=5.0),)

    with pytest.raises(ValueError, match='must default to 0'):
        jxm.Model([BadDynamicTrace()])


def test_reset_preserves_read_only_state_fields_across_a_stochastic_macro_step(key):
    # The macro-step trace reset touches ONLY declared trace fields, so an ordinary read-only field
    # (fixed by the initial condition, no writer) is preserved across a stochastic macro-step - the
    # reset is not a blanket zeroing of anything a stochastic step reads.
    model = jxm.Model(
        [Flip()]
    )  # Flip READS prob; nothing writes it -> prob is a read-only constant
    s0 = _seed(model, capacity=4, n_live=3)
    s0 = s0.set('prob', jnp.full_like(s0.radius, 0.42))

    s1 = model(s0, dt=1.0, key=key)
    assert np.allclose(np.asarray(s1.prob), np.asarray(s0.prob))  # 0.42 survives, not reset to 0
    assert np.allclose(np.asarray(s1.flip_eligible), np.asarray(s0.alive).astype(float))


def test_simulate_requires_key_for_stochastic_model(key):
    model = jxm.Model([MaybeDivide(p=jnp.array(0.5))])
    s = _seed(model, capacity=4, n_live=0)
    with pytest.raises(ValueError, match='stochastic'):
        jxm.simulate(model, s, n_steps=2)  # key=None + a stochastic step -> hard error
    final = jxm.simulate(model, s, n_steps=2, key=key)  # with a key it runs
    assert np.isclose(float(final.t), 2.0)


def test_transition_logp_zero_without_stochastic_steps():
    class Grow(jxm.SimulationStep):
        step_type = jxm.StepType.DYNAMIC

        def state_writes(self):
            return (jxm.RADIUS,)

        def __call__(self, state, *, dt, key):
            return state.deltas(radius=dt * jnp.ones_like(state.radius))

    model = jxm.Model([Grow()])
    s = _seed(model, capacity=2, n_live=0)
    assert float(jxm.transition_logp(model, s, s, 1.0)) == 0.0


def test_stochastic_step_without_replay_raises(key):
    class NoReplay(jxm.StochasticStep):
        step_type = jxm.StepType.DISCRETE
        p: jax.Array

        def trace_writes(self):
            return (jxm.StateFieldSpec('divided', shape=(), heritable=False),)

    model = jxm.Model([NoReplay(p=jnp.array(0.5))])
    s0 = _seed(model, capacity=2, n_live=0)
    with pytest.raises(NotImplementedError, match='sample_trace'):
        model(s0, dt=1.0, key=key)  # __call__ needs sample_trace + replay


def test_bernoulli_logp_matches_reference_and_is_boundary_safe():
    p = jnp.array([0.2, 0.5, 0.9])
    out = jnp.array([1.0, 0.0, 1.0])
    assert np.allclose(np.asarray(jxm.ad_utils.bernoulli_logp(out, p)), np.log([0.2, 0.5, 0.9]))
    lp = jxm.ad_utils.bernoulli_logp(jnp.array([0.0, 1.0]), jnp.array([0.0, 1.0]))
    assert np.allclose(np.asarray(lp), 0.0)
    g = jax.grad(lambda pp: jxm.ad_utils.bernoulli_logp(jnp.array([1.0]), pp).sum())(
        jnp.array([0.5])
    )
    assert np.all(np.isfinite(g)) and np.allclose(g, 2.0)  # d/dp log p = 1/p


def test_categorical_logp_matches_log_softmax():
    logits = jnp.array([0.1, 2.0, -1.0])
    onehot = jnp.array([0.0, 1.0, 0.0])
    lp = jxm.ad_utils.categorical_logp(onehot, logits)
    assert np.isclose(float(lp), float(jax.nn.log_softmax(logits)[1]))
