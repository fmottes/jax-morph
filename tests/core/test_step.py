import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_morph.core.state import RADIUS, StateFieldSpec, build_state_from_model
from jax_morph.core.step import Model, SimulationStep, StepType


def _by_name(specs):
    return {spec.name: spec for spec in specs}


class Produce(SimulationStep):
    step_type = StepType.QUASISTATIC
    name: str

    def state_writes(self):
        return (StateFieldSpec(self.name, shape=(1,)),)

    def __call__(self, state, *, dt, key):
        return state.set(self.name, state[self.name] + 1.0)


class Need(SimulationStep):
    step_type = StepType.QUASISTATIC
    name: str

    def state_reads(self):
        return (StateFieldSpec(self.name, shape=(1,)),)

    def __call__(self, state, *, dt, key):
        return state


class GrowRate(SimulationStep):
    step_type = StepType.DYNAMIC

    def state_writes(self):
        return (RADIUS,)  # a base field: use its prefilled spec

    def __call__(self, state, *, dt, key):
        return state.deltas(radius=dt * jnp.ones_like(state.radius))


def test_valid_model_builds_and_requires_merges():
    m = Model([Produce('chemical'), Need('chemical')])
    req = _by_name(m.state_requires())
    assert 'chemical' in req and req['chemical'].shape == (1,)


def test_requires_is_union_of_reads_and_writes():
    class ReadWrite(SimulationStep):
        step_type = StepType.QUASISTATIC

        def state_reads(self):
            return (StateFieldSpec('chemical', shape=(1,)),)

        def state_writes(self):
            return (StateFieldSpec('signal', shape=()),)

    req = _by_name(ReadWrite().state_requires())
    assert set(req) == {'chemical', 'signal'}
    assert req['chemical'].shape == (1,) and req['signal'].shape == ()


def test_read_only_field_is_allowed_and_allocated():
    # a field only ever read - never written by any step - is valid: it is allocated from its read
    # spec and can be fixed by the initial condition (or left constant). The model does not require
    # a writer for it.
    model = Model([Need('chemical')])  # nobody writes 'chemical'
    assert 'chemical' in _by_name(model.state_requires())
    s = build_state_from_model(model).init_empty(capacity=3, n_space_dim=2, n_types=1)
    assert s['chemical'].shape == (3, 1)  # allocated (to default), never written


def test_feedback_order_is_allowed():
    # read a field a *later* step writes: valid - it holds last macro-step's value
    Model([Need('chemical'), Produce('chemical')])


def test_two_quasistatic_writers_conflict():
    with pytest.raises(ValueError, match='written by'):
        Model([Produce('chemical'), Produce('chemical')])


def test_two_dynamic_writers_accumulate_ok():
    Model([GrowRate(), GrowRate()])  # dynamic writers may share a field: no error


class SetRadius(SimulationStep):
    step_type = StepType.QUASISTATIC

    def state_writes(self):
        return (RADIUS,)

    def __call__(self, state, *, dt, key):
        return state.set('radius', state.radius)


def test_quasistatic_and_dynamic_writer_of_same_field_conflict_both_orders():
    # a field cannot be both quasistatic (single-writer, instantaneous) and dynamic (accumulated),
    # in EITHER declaration order
    with pytest.raises(ValueError, match='dynamic'):
        Model([SetRadius(), GrowRate()])  # quasistatic then dynamic
    with pytest.raises(ValueError, match='dynamic'):
        Model([GrowRate(), SetRadius()])  # dynamic then quasistatic


def test_global_scope_dynamic_field_accumulates_and_is_not_alive_masked():
    # a dynamic writer of a GLOBAL-scope field accumulates across writers but is NOT alive-masked
    # (the alive mask applies to cell-scope totals only), so it grows even with zero live cells.
    counter = StateFieldSpec('counter', shape=(), scope='global')

    class Bump(SimulationStep):
        step_type = StepType.DYNAMIC

        def state_writes(self):
            return (counter,)

        def __call__(self, state, *, dt, key):
            return state.deltas(counter=dt * jnp.ones(()))

    model = Model([Bump(), Bump()])  # two dynamic writers of one global field
    s0 = build_state_from_model(model).init_empty(capacity=4, n_space_dim=2, n_types=1)
    assert int(s0.alive.sum()) == 0  # no live cells
    s1 = model(s0, dt=0.5, key=jax.random.PRNGKey(0))
    assert np.isclose(float(s1.counter), 1.0)  # 2 writers * 0.5, unmasked despite zero live cells


def test_model_requires_unions_and_dedups():
    # 'chemical' is written by Produce and read by Need with the *identical* spec (deduped to one);
    # 'signal' is a distinct field kept in the union.
    req = _by_name(
        Model([Produce('chemical'), Produce('signal'), Need('chemical')]).state_requires()
    )
    assert set(req) == {'chemical', 'signal'}  # union; 'chemical' appears once (deduped)
    assert req['chemical'].shape == (1,) and req['signal'].shape == (1,)


def test_inconsistent_spec_across_steps_errors():
    class ProduceWide(SimulationStep):
        step_type = StepType.QUASISTATIC

        def state_writes(self):
            return (StateFieldSpec('chemical', shape=(3,)),)

        def __call__(self, state, *, dt, key):
            return state

    # Need reads chemical as (1,), ProduceWide writes it as (3,): no write conflict, but the
    # cross-step spec merge rejects the mismatch.
    with pytest.raises(ValueError, match='conflicting specs'):
        Model([Need('chemical'), ProduceWide()])
