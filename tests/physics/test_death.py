"""Death: discrete stochastic removal with persistent lineage recording."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import jax_morph.physics as physics
from jax_morph.core.simulate import simulate, transition_logp
from jax_morph.core.state import build_state_from_model
from jax_morph.core.step import Model, check_stochastic_step
from jax_morph.physics.death import DEATH_RATE, Death
from jax_morph.physics.division import Division, reconstruct_lineage


def _death_state(capacity, n_seed, *, model=None):
    model = Death() if model is None else model
    state = build_state_from_model(model).init_empty(capacity=capacity, n_space_dim=2, n_types=1)
    return state.update(
        alive=state.alive.at[:n_seed].set(True),
        radius=state.radius.at[:n_seed].set(0.5),
        celltype=state.celltype.at[:n_seed, 0].set(1.0),
    )


def test_death_removes_alive_cells(key):
    state = _death_state(capacity=6, n_seed=4)
    state = state.set('death_rate', jnp.full((6,), 50.0))

    out = Death()(state, dt=1.0, key=key)

    assert int(out.alive.sum()) == 0
    assert np.array_equal(np.asarray(out.death), np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0]))


def test_death_is_exported_from_physics():
    assert physics.Death is Death
    assert physics.DEATH_RATE is DEATH_RATE


def test_death_is_dt_consistent(key):
    lam, n_cells = 0.5, 4096
    state = _death_state(capacity=n_cells, n_seed=n_cells)
    state = state.set('death_rate', jnp.full((n_cells,), lam))
    for dt in (0.1, 0.2):
        out = Death()(state, dt=dt, key=key)
        killed_fraction = 1.0 - int(out.alive.sum()) / n_cells
        assert np.isclose(killed_fraction, 1.0 - np.exp(-lam * dt), atol=0.03)


def test_death_records_persistent_flag_and_trace(key):
    step = Death()
    state = _death_state(capacity=6, n_seed=3)
    state = state.set('death_rate', jnp.array([50.0, 50.0, 50.0, 50.0, 50.0, 50.0]))

    out = step(state, dt=1.0, key=key)
    trace = step.trace_from_state(out)

    assert np.array_equal(np.asarray(out.death), np.asarray(trace['died']))
    assert np.array_equal(
        np.asarray(trace['die_eligible']), np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    )


def test_death_trace_round_trips(key):
    state = _death_state(capacity=4, n_seed=2)
    state = state.set('death_rate', jnp.full((4,), 1.0))
    check_stochastic_step(Death(), state, dt=1.0, key=key)


def test_logp_scores_only_eligible_cells(key):
    step = Death()
    state = _death_state(capacity=6, n_seed=3)
    state = state.set('death_rate', jnp.full((6,), 0.4))
    out = step(state, dt=1.0, key=key)
    trace = step.trace_from_state(out)
    p = 1.0 - np.exp(-0.4)
    died = np.asarray(trace['died'])
    eligible = np.asarray(trace['die_eligible'])
    expected = np.sum((died * np.log(p) + (1.0 - died) * np.log(1.0 - p)) * eligible)

    assert np.isclose(float(step.logp(state, trace, 1.0)), expected, atol=1e-5)
    assert np.allclose(eligible[3:], 0.0)


def test_logp_gradient_reaches_the_death_rate():
    step = Death()
    state = _death_state(capacity=2, n_seed=2)
    trace = {
        'died': jnp.array([1.0, 0.0]),
        'die_eligible': jnp.array([1.0, 1.0]),
    }

    def logp(rate):
        return step.logp(state.set('death_rate', rate), trace, 1.0)

    grad = jax.grad(logp)(jnp.array([0.5, 0.5]))
    assert np.all(np.isfinite(np.asarray(grad)))
    assert float(grad[0]) > 0.0
    assert float(grad[1]) < 0.0


def test_physical_death_effect_has_no_pathwise_gradient(key):
    model = Model([Death()])
    state = _death_state(capacity=16, n_seed=16, model=model)

    def physical_loss(rate):
        out = simulate(model, state.set('death_rate', rate), n_steps=1, dt=1.0, key=key)
        return jnp.sum(out.death)

    def trace_loss(rate):
        out = simulate(model, state.set('death_rate', rate), n_steps=1, dt=1.0, key=key)
        return jnp.sum(out.died)

    rate = jnp.full((16,), 1.0)
    physical_grad = eqx.filter_grad(physical_loss)(rate)
    trace_grad = eqx.filter_grad(trace_loss)(rate)
    assert np.array_equal(np.asarray(physical_grad), np.zeros((16,)))
    assert float(jnp.max(jnp.abs(trace_grad))) > 0.0


def test_zero_death_rate_is_exactly_off(key):
    step = Death()
    state = _death_state(capacity=8, n_seed=4)
    state = state.set('death_rate', jnp.zeros((8,)))

    assert np.array_equal(np.asarray(step._dist(state, 1.0)), np.zeros((8,)))
    out = step(state, dt=1.0, key=key)
    trace = step.trace_from_state(out)
    assert np.array_equal(np.asarray(trace['died']), np.zeros((8,)))
    assert float(step.logp(state, trace, 1.0)) == 0.0


def test_all_dead_is_nan_free(key):
    step = Death()
    state = _death_state(capacity=4, n_seed=0)
    state = state.set('death_rate', jnp.full((4,), 1.0))

    out = step(state, dt=1.0, key=key)

    assert int(out.alive.sum()) == 0
    assert np.array_equal(np.asarray(out.death), np.zeros((4,)))


def test_replay_is_deterministic_given_the_recorded_trace(key):
    step = Death()
    state = _death_state(capacity=6, n_seed=3)
    state = state.set('death_rate', jnp.full((6,), 1.0))
    out = step(state, dt=1.0, key=key)
    trace = step.trace_from_state(out)
    replay_a = step.replay(state, trace, dt=1.0, pathwise=False)
    replay_b = step.replay(state, trace, dt=1.0, pathwise=False)

    assert np.array_equal(np.asarray(replay_a.alive), np.asarray(replay_b.alive))
    assert np.array_equal(np.asarray(replay_a.alive), np.asarray(out.alive))
    assert np.array_equal(np.asarray(replay_a.death), np.asarray(out.death))


def test_transition_logp_is_finite_and_deterministic(key):
    model = Model([Death()])
    state = _death_state(capacity=6, n_seed=3, model=model)
    state = state.set('death_rate', jnp.full((6,), 1.0))
    next_state = model(state, dt=1.0, key=key)

    lp_a = float(transition_logp(model, state, next_state, dt=1.0))
    lp_b = float(transition_logp(model, state, next_state, dt=1.0))
    assert np.isfinite(lp_a) and lp_a == lp_b


def test_reconstruct_lineage_annotates_death_step():
    born = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]], dtype=float)
    mother = np.array([[-1, -1, -1], [-1, 0, -1], [1, -1, -1]], dtype=int)
    alive = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0]], dtype=bool)
    death = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)

    nodes = reconstruct_lineage(born, mother, alive, death=death)

    assert nodes == [
        {'id': 0, 'parent': None, 'slot': 0, 'birth_step': 0, 'death_step': 1},
        {'id': 1, 'parent': 0, 'slot': 1, 'birth_step': 1, 'death_step': 2},
        {'id': 2, 'parent': 1, 'slot': 0, 'birth_step': 2, 'death_step': None},
    ]
    without_death = reconstruct_lineage(born, mother, alive)
    assert all(node['death_step'] is None for node in without_death)


def test_division_then_death_lineage_integration(key):
    model = Model([Division(n_space_dim=2), Death()])
    state = _death_state(capacity=2, n_seed=1, model=model)
    state = state.update(
        division_rate=state.division_rate.at[0].set(50.0),
        death_rate=jnp.zeros((2,)),
    )

    first = simulate(model, state, n_steps=1, dt=1.0, key=key, history=True)
    state = jax.tree_util.tree_map(lambda value: value[-1], first).update(
        division_rate=jnp.zeros((2,)),
        death_rate=jnp.array([50.0, 0.0]),
    )
    second = simulate(model, state, n_steps=1, dt=1.0, key=key, history=True)
    state = jax.tree_util.tree_map(lambda value: value[-1], second).update(
        division_rate=jnp.array([0.0, 50.0]),
        death_rate=jnp.zeros((2,)),
    )
    third = simulate(model, state, n_steps=1, dt=1.0, key=key, history=True)
    history = jax.tree_util.tree_map(
        lambda a, b, c: jnp.concatenate((a, b[1:], c[1:]), axis=0), first, second, third
    )

    nodes = reconstruct_lineage(history.born, history.mother, history.alive, death=history.death)

    assert nodes[0]['death_step'] == 2
    assert nodes[1]['parent'] == 0 and nodes[1]['birth_step'] == 1
    assert nodes[2]['parent'] == 1 and nodes[2]['slot'] == 0 and nodes[2]['birth_step'] == 3
    inferred = (history.alive[:-1] | (history.born[1:] > 0.5)) & ~history.alive[1:]
    assert np.array_equal(np.asarray(history.death[1:] > 0.5), np.asarray(inferred))


def test_division_then_death_can_kill_a_newborn_in_one_macro_step(key):
    model = Model([Division(n_space_dim=2), Death()])
    state = _death_state(capacity=2, n_seed=1, model=model)
    state = state.update(
        division_rate=state.division_rate.at[0].set(50.0),
        death_rate=state.death_rate.at[0].set(50.0),
    )

    history = simulate(model, state, n_steps=1, dt=1.0, key=key, history=True)
    nodes = reconstruct_lineage(history.born, history.mother, history.alive, death=history.death)

    assert np.array_equal(np.asarray(history.born[1] > 0.5), np.array([False, True]))
    assert np.array_equal(np.asarray(history.death[1] > 0.5), np.array([True, True]))
    assert np.array_equal(np.asarray(history.alive[1]), np.array([False, False]))
    assert nodes == [
        {'id': 0, 'parent': None, 'slot': 0, 'birth_step': 0, 'death_step': 1},
        {'id': 1, 'parent': 0, 'slot': 1, 'birth_step': 1, 'death_step': 1},
    ]
