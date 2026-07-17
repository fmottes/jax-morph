"""Acceptance tests for live-carry trajectory trace scoring."""

import itertools

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import jax_morph as jxm

X = jxm.StateFieldSpec('x', shape=())
RATE = jxm.StateFieldSpec('rate', shape=())
U = jxm.StateFieldSpec('u', shape=(), heritable=False)


def _gaussian_logp(x, mean, std):
    z = (x - mean) / std
    return -0.5 * z * z - jnp.log(std) - 0.5 * jnp.log(2.0 * jnp.pi)


class Kick(jxm.StochasticStep):
    """Reparameterized dynamic kick used to test cross-step gradient barriers."""

    step_type = jxm.StepType.DYNAMIC
    log_std: jax.Array
    mean_coeff: float = eqx.field(static=True, default=0.0)

    def state_writes(self):
        return (U,)

    def trace_writes(self):
        return (
            jxm.StateFieldSpec('kick_xi', shape=(), heritable=False),
            jxm.StateFieldSpec('kick_dx', shape=(), heritable=False),
        )

    def _dist(self, state, dt):
        mean = self.mean_coeff * state.u
        std = jnp.exp(self.log_std) * jnp.ones_like(state.u)
        return mean, std

    def sample_trace(self, state, *, dt, key):
        return {'kick_xi': jax.random.normal(key, state.u.shape)}

    def replay(self, state, trace, *, dt, pathwise):
        if pathwise:
            mean, std = self._dist(state, dt)
            dx = mean + std * trace['kick_xi']
        else:
            dx = trace['kick_dx']
        return state.deltas(u=dx, kick_xi=trace['kick_xi'], kick_dx=dx)

    def logp(self, state, trace, dt):
        mean, std = self._dist(state, dt)
        return jnp.sum(_gaussian_logp(trace['kick_dx'], mean, std) * state.alive.astype(std.dtype))


class DivU(jxm.StochasticStep):
    """Discrete decision whose probability depends on the kicked field."""

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
        action = jxm.ad_utils.sample_bernoulli_st(key, self._dist(state, dt))
        eligible = state.alive.astype(action.dtype)
        return {'divu': action * eligible, 'divu_eligible': eligible}

    def replay(self, state, trace, *, dt, pathwise):
        return state.update(divu=trace['divu'], divu_eligible=trace['divu_eligible'])

    def logp(self, state, trace, dt):
        return jnp.sum(
            jxm.ad_utils.bernoulli_logp(trace['divu'], self._dist(state, dt))
            * trace['divu_eligible']
        )


class Drift(jxm.SimulationStep):
    """Integrate ``dx/dt = theta``."""

    step_type = jxm.StepType.DYNAMIC
    theta: jax.Array

    def state_writes(self):
        return (X,)

    def __call__(self, state, *, dt, key):
        return state.deltas(x=dt * self.theta * jnp.ones_like(state.x))


class SetRate(jxm.SimulationStep):
    """Set an exponential hazard from the current dynamic state."""

    step_type = jxm.StepType.QUASISTATIC

    def state_reads(self):
        return (X,)

    def state_writes(self):
        return (RATE,)

    def __call__(self, state, *, dt, key):
        return state.set('rate', jnp.exp(state.x))


class LaggedHazard(jxm.StochasticStep):
    """Draw a Bernoulli event from a quasistatic pre-dynamic hazard."""

    step_type = jxm.StepType.DISCRETE
    score_by_default: bool = eqx.field(static=True, default=True, kw_only=True)

    def state_reads(self):
        return (RATE,)

    def trace_writes(self):
        return (
            jxm.StateFieldSpec('lagged', shape=(), heritable=False),
            jxm.StateFieldSpec('lagged_eligible', shape=(), heritable=False),
        )

    def _dist(self, state, dt):
        return -jnp.expm1(-state.rate * dt)

    def sample_trace(self, state, *, dt, key):
        action = jxm.ad_utils.sample_bernoulli_st(key, self._dist(state, dt))
        eligible = state.alive.astype(action.dtype)
        return {'lagged': action * eligible, 'lagged_eligible': eligible}

    def replay(self, state, trace, *, dt, pathwise):
        return state.update(lagged=trace['lagged'], lagged_eligible=trace['lagged_eligible'])

    def logp(self, state, trace, dt):
        return jnp.sum(
            jxm.ad_utils.bernoulli_logp(trace['lagged'], self._dist(state, dt))
            * trace['lagged_eligible']
        )


class CurrentHazard(jxm.StochasticStep):
    """Draw from the post-dynamic state, representing the opposite split ordering."""

    step_type = jxm.StepType.DISCRETE

    def state_reads(self):
        return (X,)

    def trace_writes(self):
        return (
            jxm.StateFieldSpec('current', shape=(), heritable=False),
            jxm.StateFieldSpec('current_eligible', shape=(), heritable=False),
        )

    def _dist(self, state, dt):
        return -jnp.expm1(-jnp.exp(state.x) * dt)

    def sample_trace(self, state, *, dt, key):
        action = jxm.ad_utils.sample_bernoulli_st(key, self._dist(state, dt))
        eligible = state.alive.astype(action.dtype)
        return {'current': action * eligible, 'current_eligible': eligible}

    def replay(self, state, trace, *, dt, pathwise):
        return state.update(current=trace['current'], current_eligible=trace['current_eligible'])

    def logp(self, state, trace, dt):
        return jnp.sum(
            jxm.ad_utils.bernoulli_logp(trace['current'], self._dist(state, dt))
            * trace['current_eligible']
        )


class Toggle(jxm.StochasticStep):
    """A finite binary Markov model used to enumerate importance weights."""

    step_type = jxm.StepType.DISCRETE
    b: jax.Array
    coupling: float = eqx.field(static=True, default=0.7)

    def state_reads(self):
        return (X,)

    def state_writes(self):
        return (X,)

    def trace_writes(self):
        return (
            jxm.StateFieldSpec('toggle', shape=(), heritable=False),
            jxm.StateFieldSpec('toggle_eligible', shape=(), heritable=False),
        )

    def _dist(self, state, dt):
        return jax.nn.sigmoid(self.b + self.coupling * state.x)

    def sample_trace(self, state, *, dt, key):
        action = jxm.ad_utils.sample_bernoulli_st(key, self._dist(state, dt))
        eligible = state.alive.astype(action.dtype)
        return {'toggle': action * eligible, 'toggle_eligible': eligible}

    def replay(self, state, trace, *, dt, pathwise):
        return state.update(
            x=trace['toggle'],
            toggle=trace['toggle'],
            toggle_eligible=trace['toggle_eligible'],
        )

    def logp(self, state, trace, dt):
        return jnp.sum(
            jxm.ad_utils.bernoulli_logp(trace['toggle'], self._dist(state, dt))
            * trace['toggle_eligible']
        )


class BitEffect(jxm.StochasticStep):
    """Frozen bit with a conditionally differentiable continuous physical effect."""

    step_type = jxm.StepType.DISCRETE
    beta: jax.Array
    p: jax.Array

    def state_reads(self):
        return (X,)

    def state_writes(self):
        return (X,)

    def trace_writes(self):
        return (
            jxm.StateFieldSpec('bit', shape=(), heritable=False),
            jxm.StateFieldSpec('bit_eligible', shape=(), heritable=False),
        )

    def _dist(self, state, dt):
        return self.p * jnp.ones_like(state.x)

    def sample_trace(self, state, *, dt, key):
        bit = jxm.ad_utils.sample_bernoulli_st(key, self._dist(state, dt))
        eligible = state.alive.astype(bit.dtype)
        return {'bit': bit * eligible, 'bit_eligible': eligible}

    def replay(self, state, trace, *, dt, pathwise):
        x = state.x + trace['bit'] * self.beta * state.x
        return state.update(x=x, bit=trace['bit'], bit_eligible=trace['bit_eligible'])

    def logp(self, state, trace, dt):
        return jnp.sum(
            jxm.ad_utils.bernoulli_logp(trace['bit'], self._dist(state, dt)) * trace['bit_eligible']
        )


def _seed(model, capacity, n_live):
    state = jxm.build_state_from_model(model).init_empty(
        capacity=capacity, n_space_dim=2, n_types=1
    )
    return state.set('alive', state.alive.at[:n_live].set(True))


def _frame(history, index):
    return jax.tree_util.tree_map(lambda x: x[index], history)


def _tail(history):
    return jax.tree_util.tree_map(lambda x: x[1:], history)


def _force_trace(history, action_name, actions):
    actions = jnp.asarray(actions, dtype=history[action_name].dtype)
    tail = history[action_name][1:]
    actions = actions.reshape((actions.shape[0],) + (1,) * (tail.ndim - 1))
    actions = jnp.broadcast_to(actions, tail.shape)
    eligible = history.alive[1:].astype(actions.dtype)
    return history.update(
        **{
            action_name: history[action_name].at[1:].set(actions * eligible),
            f'{action_name}_eligible': history[f'{action_name}_eligible'].at[1:].set(eligible),
        }
    )


def _lagged_model(theta=0.4):
    return jxm.Model([SetRate(), Drift(theta=jnp.asarray(theta)), LaggedHazard()])


def test_scoring_drivers_are_public_and_model_logp_is_removed():
    assert 'trajectory_logp' in jxm.__all__
    assert 'transition_logp' in jxm.__all__
    assert not hasattr(jxm.Model, 'logp')


def test_lagged_quasistatic_gradient_crosses_macro_step_boundary(key):
    dt = 0.2
    model = _lagged_model()
    s0 = _seed(model, capacity=1, n_live=1)
    history = jxm.simulate(model, s0, 3, dt, key, history=True)
    history = _force_trace(history, 'lagged', [0.0, 1.0, 0.0])

    grad = eqx.filter_grad(lambda m: jxm.trajectory_logp(m, history, dt).sum())(model)
    conditional_grad = eqx.filter_grad(
        lambda m: sum(
            jxm.transition_logp(m, _frame(history, i), _frame(history, i + 1), dt) for i in range(3)
        )
    )(model)

    x = np.arange(3) * dt * float(model.steps[1].theta)
    rate = np.exp(x)
    probability = 1.0 - np.exp(-rate * dt)
    action = np.array([0.0, 1.0, 0.0])
    dq_dx = dt * rate * np.exp(-rate * dt)
    analytic = np.sum(
        (action / probability - (1.0 - action) / (1.0 - probability)) * dq_dx * np.arange(3) * dt
    )
    assert np.isclose(float(grad.steps[1].theta), analytic, rtol=2e-5)
    assert not np.isclose(float(grad.steps[1].theta), 0.0)
    assert np.isclose(float(conditional_grad.steps[1].theta), 0.0)


def test_splitting_orders_converge_to_same_continuous_time_gradient(key):
    theta, horizon = 0.35, 1.0
    exact = -((np.exp(theta * horizon) * (theta * horizon - 1.0) + 1.0) / theta**2)
    errors = []
    for n_steps in (10, 40, 160):
        dt = horizon / n_steps
        lagged = _lagged_model(theta)
        current = jxm.Model([Drift(theta=jnp.asarray(theta)), CurrentHazard()])
        s_lag = _seed(lagged, 1, 1)
        s_cur = _seed(current, 1, 1)
        h_lag = _force_trace(
            jxm.simulate(lagged, s_lag, n_steps, dt, key, history=True),
            'lagged',
            np.zeros(n_steps),
        )
        h_cur = _force_trace(
            jxm.simulate(current, s_cur, n_steps, dt, key, history=True),
            'current',
            np.zeros(n_steps),
        )
        g_lag = (
            eqx.filter_grad(lambda m, h=h_lag, step=dt: jxm.trajectory_logp(m, h, step).sum())(
                lagged
            )
            .steps[1]
            .theta
        )
        g_cur = (
            eqx.filter_grad(lambda m, h=h_cur, step=dt: jxm.trajectory_logp(m, h, step).sum())(
                current
            )
            .steps[0]
            .theta
        )
        errors.append(
            (abs(float(g_lag) - exact), abs(float(g_cur) - exact), abs(float(g_lag - g_cur)))
        )

    assert errors[-1][0] < errors[0][0] / 5.0
    assert errors[-1][1] < errors[0][1] / 5.0
    assert errors[-1][2] < errors[0][2] / 5.0


def test_same_parameter_terms_match_conditional_scores_but_gradients_differ(key):
    model = _lagged_model()
    s0 = _seed(model, 1, 1)
    history = jxm.simulate(model, s0, 4, 0.25, key, history=True)
    terms = jxm.trajectory_logp(model, history, 0.25)
    conditional = jnp.stack(
        [
            jxm.transition_logp(model, _frame(history, i), _frame(history, i + 1), 0.25)
            for i in range(4)
        ]
    )
    assert np.allclose(np.asarray(terms), np.asarray(conditional), rtol=1e-6)

    live_grad = (
        eqx.filter_grad(lambda m: jxm.trajectory_logp(m, history, 0.25).sum())(model).steps[1].theta
    )
    pair_grad = (
        eqx.filter_grad(
            lambda m: sum(
                jxm.transition_logp(m, _frame(history, i), _frame(history, i + 1), 0.25)
                for i in range(4)
            )
        )(model)
        .steps[1]
        .theta
    )
    assert not np.isclose(float(live_grad), float(pair_grad))
    assert np.isclose(float(pair_grad), 0.0)


def test_on_policy_reconstructed_carry_equals_recorded_frames(key):
    model = _lagged_model()
    s0 = _seed(model, 2, 1)
    history = jxm.simulate(model, s0, 5, 0.1, key, history=True)
    scored = model._resolve_score(None)

    def replay(carry, recorded):
        next_state, _ = model._replay_transition(carry, recorded, 0.1, scored)
        return next_state, next_state

    _, reconstructed = jax.lax.scan(
        replay, jax.lax.stop_gradient(_frame(history, 0)), _tail(history)
    )
    for actual, expected in zip(
        jax.tree_util.tree_leaves(reconstructed),
        jax.tree_util.tree_leaves(_tail(history)),
        strict=True,
    ):
        assert np.allclose(np.asarray(actual), np.asarray(expected), rtol=1e-6, atol=1e-7)


def test_time_advances_and_dynamic_trace_values_stay_per_step(key):
    kick = Kick(log_std=jnp.asarray(np.log(0.4)), score_by_default=True)
    model = jxm.Model([kick])
    s0 = _seed(model, 2, 1)
    history = jxm.simulate(model, s0, 4, 0.3, key, history=True)
    scored = model._resolve_score('all')

    def replay(carry, recorded):
        next_state, term = model._replay_transition(carry, recorded, 0.3, scored)
        return next_state, (next_state, term)

    _, (reconstructed, terms) = jax.lax.scan(replay, _frame(history, 0), _tail(history))
    assert np.allclose(np.asarray(reconstructed.t), np.arange(1, 5) * 0.3)
    assert np.allclose(np.asarray(reconstructed.kick_dx), np.asarray(history.kick_dx[1:]))
    assert np.all(np.isfinite(np.asarray(terms)))


def test_unscored_reparameterized_path_remains_live_across_steps(key):
    model = jxm.Model(
        [
            Kick(
                log_std=jnp.asarray(np.log(0.6)),
                mean_coeff=0.0,
                score_by_default=False,
            ),
            DivU(b=jnp.asarray(-0.2), c=1.3, score_by_default=True),
        ]
    )
    s0 = _seed(model, 3, 2)
    history = jxm.simulate(model, s0, 4, 1.0, key, history=True)
    grad = eqx.filter_grad(lambda m: jxm.trajectory_logp(m, history).sum())(model)
    pair_grad = eqx.filter_grad(
        lambda m: sum(
            jxm.transition_logp(m, _frame(history, i), _frame(history, i + 1)) for i in range(4)
        )
    )(model)
    assert not np.isclose(float(grad.steps[0].log_std), 0.0)
    assert not np.isclose(float(grad.steps[0].log_std), float(pair_grad.steps[0].log_std))


def test_scored_reparameterized_action_is_frozen_for_later_terms(key):
    model = jxm.Model(
        [
            Kick(log_std=jnp.asarray(np.log(0.6)), score_by_default=True),
            DivU(b=jnp.asarray(0.1), c=1.1, score_by_default=True),
        ]
    )
    s0 = _seed(model, 3, 2)
    history = jxm.simulate(model, s0, 4, 1.0, key, history=True)
    kick_only = eqx.filter_grad(lambda m: jxm.trajectory_logp(m, history, score=[0]).sum())(model)
    both = eqx.filter_grad(lambda m: jxm.trajectory_logp(m, history, score='all').sum())(model)
    assert np.isclose(float(kick_only.steps[0].log_std), float(both.steps[0].log_std))


def test_frozen_discrete_event_retains_conditional_continuous_effect(key):
    model = jxm.Model(
        [SetRate(), BitEffect(beta=jnp.asarray(0.2), p=jnp.asarray(0.4)), LaggedHazard()]
    )
    s0 = _seed(model, 1, 1)
    s0 = s0.set('x', jnp.ones_like(s0.x))
    history = jxm.simulate(model, s0, 3, 0.1, key, history=True)
    history = _force_trace(_force_trace(history, 'bit', [1, 1, 1]), 'lagged', [1, 1, 1])
    all_grad = eqx.filter_grad(lambda m: jxm.trajectory_logp(m, history, 0.1, score='all').sum())(
        model
    )
    bit_grad = eqx.filter_grad(lambda m: jxm.trajectory_logp(m, history, 0.1, score=[1]).sum())(
        model
    )
    assert not np.isclose(float(all_grad.steps[1].beta), 0.0)
    assert np.isclose(float(all_grad.steps[1].p), float(bit_grad.steps[1].p))


def test_shifted_parameter_mle_gradient_matches_finite_difference(key):
    reference = _lagged_model(theta=0.3)
    candidate = _lagged_model(theta=0.47)
    s0 = _seed(reference, 1, 1)
    history = jxm.simulate(reference, s0, 5, 0.2, key, history=True)

    def objective(theta):
        model = _lagged_model(theta)
        return jxm.trajectory_logp(model, history, 0.2, score='all').sum()

    analytic = jax.grad(objective)(candidate.steps[1].theta)
    epsilon = 1e-3
    finite = (objective(0.47 + epsilon) - objective(0.47 - epsilon)) / (2 * epsilon)
    assert np.isclose(float(analytic), float(finite), rtol=3e-3, atol=3e-4)


def test_enumerated_importance_weights_reproduce_target_expectation(key):
    old = jxm.Model([Toggle(b=jnp.asarray(-0.4))])
    new = jxm.Model([Toggle(b=jnp.asarray(0.35))])
    s0 = _seed(old, 1, 1)
    template = jxm.simulate(old, s0, 2, 1.0, key, history=True)
    old_probabilities = []
    target_probabilities = []
    outcomes = []
    weights = []
    for sequence in itertools.product((0.0, 1.0), repeat=2):
        history = _force_trace(template, 'toggle', sequence)
        lp_old = jxm.trajectory_logp(old, history, score='all').sum()
        lp_new = jxm.trajectory_logp(new, history, score='all').sum()
        old_p = float(jnp.exp(lp_old))
        new_p = float(jnp.exp(lp_new))
        old_probabilities.append(old_p)
        target_probabilities.append(new_p)
        outcomes.append(sequence[-1])
        weights.append(float(jnp.exp(lp_new - lp_old)))

    old_probabilities = np.asarray(old_probabilities)
    target_probabilities = np.asarray(target_probabilities)
    outcomes = np.asarray(outcomes)
    weights = np.asarray(weights)
    assert np.isclose(old_probabilities.sum(), 1.0)
    assert np.isclose(target_probabilities.sum(), 1.0)
    assert np.isclose(
        np.sum(old_probabilities * weights * outcomes),
        np.sum(target_probabilities * outcomes),
    )


def test_score_selection_is_preserved_through_scan(key):
    model = jxm.Model(
        [
            Kick(log_std=jnp.asarray(np.log(0.7)), score_by_default=False),
            DivU(b=jnp.asarray(0.2), score_by_default=True),
        ]
    )
    s0 = _seed(model, 2, 1)
    history = jxm.simulate(model, s0, 3, 1.0, key, history=True)
    kick = jxm.trajectory_logp(model, history, score=[0])
    div = jxm.trajectory_logp(model, history, score=[1])
    all_terms = jxm.trajectory_logp(model, history, score='all')
    default = jxm.trajectory_logp(model, history)
    assert np.allclose(np.asarray(all_terms), np.asarray(kick + div))
    assert np.allclose(np.asarray(default), np.asarray(div))


def test_jit_vmap_and_checkpoint_match_values_and_gradients(key):
    model = _lagged_model()
    s0 = _seed(model, 1, 1)
    history = jxm.simulate(model, s0, 4, 0.2, key, history=True)
    eager = jxm.trajectory_logp(model, history, 0.2)
    jitted = eqx.filter_jit(jxm.trajectory_logp)(model, history, 0.2)
    checkpointed = jxm.trajectory_logp(model, history, 0.2, checkpoint=True)
    assert np.allclose(np.asarray(eager), np.asarray(jitted))
    assert np.allclose(np.asarray(eager), np.asarray(checkpointed))

    grad = eqx.filter_grad(lambda m: jxm.trajectory_logp(m, history, 0.2).sum())(model)
    checkpoint_grad = eqx.filter_grad(
        lambda m: jxm.trajectory_logp(m, history, 0.2, checkpoint=True).sum()
    )(model)
    assert np.isclose(float(grad.steps[1].theta), float(checkpoint_grad.steps[1].theta))

    batched = jax.tree_util.tree_map(lambda x: jnp.stack([x, x]), history)
    scores = jax.vmap(lambda h: jxm.trajectory_logp(model, h, 0.2))(batched)
    assert scores.shape == (2, 4)
    assert np.allclose(np.asarray(scores[0]), np.asarray(eager))


def test_deterministic_trajectory_has_shaped_zero_terms(key):
    model = jxm.Model([Drift(theta=jnp.asarray(0.2))])
    s0 = _seed(model, 2, 1)
    history = jxm.simulate(model, s0, 3, 0.1, key, history=True)
    terms = jxm.trajectory_logp(model, history, 0.1)
    assert terms.shape == (3,)
    assert np.allclose(np.asarray(terms), 0.0)


def test_zero_step_history_returns_empty_terms(key):
    model = _lagged_model()
    s0 = _seed(model, 1, 1)
    empty = jxm.simulate(model, s0, 0, 0.1, key, history=True)
    empty_terms = jxm.trajectory_logp(model, empty, 0.1, score='all')
    assert empty_terms.shape == (0,)


def test_array_dt_retains_score_derivative(key):
    model = _lagged_model(theta=0.3)
    s0 = _seed(model, 1, 1)
    dt = jnp.asarray(0.2)
    history = jxm.simulate(model, s0, 4, dt, key, history=True)

    def objective(step_size):
        return jxm.trajectory_logp(model, history, step_size).sum()

    analytic = jax.grad(objective)(dt)
    epsilon = 1e-3
    finite = (objective(dt + epsilon) - objective(dt - epsilon)) / (2 * epsilon)
    assert np.isfinite(float(analytic))
    assert np.isclose(float(analytic), float(finite), rtol=5e-3, atol=5e-3)
