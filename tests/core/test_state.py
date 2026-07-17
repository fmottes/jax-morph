import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_morph.core import state as st


class _FakeModel:
    """Stand-in for a Model: requires a 'chemical' field of width 2 (base fields are the state's)."""

    def state_requires(self):
        return (st.StateFieldSpec('chemical', shape=(2,)),)


def test_base_specs_and_spec_override():
    assert {spec.name for spec in st.BASE_SPECS} == {'position', 'radius', 'celltype', 'alive', 't'}
    assert st.POSITION.name == 'position'  # each spec carries its own name
    assert st.ALIVE.dtype is bool
    assert st.TIME.scope == 'global'
    # a prefilled base spec is customisable by calling it (returns a copy)
    assert st.POSITION.heritable is True
    assert st.POSITION(heritable=False).heritable is False
    assert st.POSITION.heritable is True  # original unchanged


def test_merge_specs_conflict():
    with pytest.raises(ValueError):
        st.merge_specs((st.StateFieldSpec('x', shape=(1,)),), (st.StateFieldSpec('x', shape=(2,)),))


def test_build_state_shapes_and_is_empty():
    s = st.build_state_from_model(_FakeModel()).init_empty(capacity=10, n_space_dim=2, n_types=3)
    assert s.position.shape == (10, 2)  # base shape filled from n_space_dim
    assert s.celltype.shape == (10, 3)  # base shape filled from n_types
    assert s.radius.shape == (10,)
    assert s['chemical'].shape == (10, 2)  # custom field allocated like any other
    # pure allocation: no cell alive, every field at its default (base and custom alike)
    assert s.alive.dtype == bool and int(s.alive.sum()) == 0
    assert np.allclose(np.asarray(s.radius), 0.0)
    assert np.allclose(np.asarray(s.celltype), 0.0)
    assert np.allclose(np.asarray(s['chemical']), 0.0)


def test_initial_condition_is_set_manually(key):
    # allocation is the constructor's job; the initial condition is the simulation's, set with
    # update() - base and custom fields identically (no first-class/second-class distinction)
    s = st.build_state_from_model(_FakeModel()).init_empty(capacity=6, n_space_dim=2, n_types=3)
    pos = jax.random.normal(key, (2, 2))
    s = s.update(
        alive=s.alive.at[:2].set(True),
        radius=s.radius.at[:2].set(0.5),
        celltype=s.celltype.at[:2, 0].set(1.0),
        position=s.position.at[:2].set(pos),
        chemical=s['chemical'].at[:2].set(1.5),
    )
    assert int(s.alive.sum()) == 2
    assert np.allclose(np.asarray(s.radius)[:2], 0.5)
    assert np.allclose(np.asarray(s.celltype)[:2].sum(axis=1), 1.0)
    assert np.allclose(np.asarray(s.position)[:2], np.asarray(pos))
    assert np.allclose(np.asarray(s['chemical'])[:2], 1.5)  # a custom field IC, same mechanism


def test_field_wise_constructor_and_field_mismatch():
    State = st.build_state_from_model(_FakeModel())
    # the plain constructor takes every field array directly (base + custom); space is optional,
    # the static schema is filled automatically
    s = State(
        position=jnp.zeros((3, 2)),
        radius=jnp.full((3,), 0.5),
        celltype=jnp.zeros((3, 1)).at[:, 0].set(1.0),
        alive=jnp.array([True, True, False]),
        t=jnp.asarray(0.0),
        chemical=jnp.ones((3, 2)),
    )
    assert int(s.alive.sum()) == 2
    assert np.allclose(np.asarray(s.radius), 0.5)
    assert np.allclose(np.asarray(s['chemical']), 1.0)  # a custom field, passed like any other
    # omitting a field (or passing an unknown one) is a clear error - shapes can't be guessed
    with pytest.raises(TypeError, match='missing fields'):
        State(position=jnp.zeros((3, 2)))


def test_set_is_functional_and_pytree():
    s = st.build_state_from_model(_FakeModel()).init_empty(capacity=4, n_space_dim=2, n_types=1)
    r0 = np.asarray(s.radius)  # snapshot (empty state: all default)
    s2 = s.set('radius', s.radius + 1.0)
    assert np.allclose(s2.radius, r0 + 1.0)
    assert np.allclose(s.radius, r0)  # original unchanged (.set is functional, not in-place)
    leaves = jax.tree_util.tree_leaves(s2)  # is a pytree
    assert all(isinstance(x, jax.Array) for x in leaves)


def test_update_replaces_multiple_fields():
    s = st.build_state_from_model(_FakeModel()).init_empty(capacity=4, n_space_dim=2, n_types=1)
    r0, c0 = np.asarray(s.radius), np.asarray(s['chemical'])
    s2 = s.update(radius=s.radius + 1.0, chemical=s['chemical'] + 2.0)
    assert np.allclose(s2.radius, r0 + 1.0)
    assert np.allclose(np.asarray(s2['chemical']), c0 + 2.0)
    assert np.allclose(s.radius, r0)  # original unchanged


def test_deltas_is_sparse_and_applies_with_apply_updates():
    import equinox as eqx

    s = st.build_state_from_model(_FakeModel()).init_empty(capacity=4, n_space_dim=2, n_types=1)
    d = s.deltas(radius=jnp.ones_like(s.radius))
    # only the named field is populated; every other field is None (an eqx "updates" tree)
    assert d.radius is not None and d.position is None and d.alive is None
    # applying it as an update adds to the named field and leaves the rest untouched
    r0 = np.asarray(s.radius)
    s2 = eqx.apply_updates(s, d)
    assert np.allclose(s2.radius, r0 + 1.0)
    assert np.allclose(np.asarray(s2.position), np.asarray(s.position))  # untouched (None) field
    # deltas() with no fields is the fully-None updates tree
    empty = s.deltas()
    assert empty.radius is None and empty.position is None


def test_global_field_and_time():
    class _GM:  # a custom layer declaring an Eulerian grid + relying on the base time field
        def state_requires(self):
            return (st.StateFieldSpec('nutrient', shape=(4, 4), scope='global'),)

    s = st.build_state_from_model(_GM()).init_empty(capacity=8, n_space_dim=2, n_types=1)
    assert s['nutrient'].shape == (4, 4)  # global: NO leading capacity axis
    assert s.position.shape == (8, 2)  # cell field: leading capacity axis
    assert float(s.t) == 0.0  # base global simulation time


def test_injectable_periodic_space():
    from jax_morph.core import geometry as geo

    s = st.build_state_from_model(_FakeModel()).init_empty(
        capacity=4, n_space_dim=2, n_types=1, space=geo.periodic(10.0)
    )
    d = s.displacement(jnp.array([9.0, 0.0]), jnp.array([1.0, 0.0]))
    assert np.allclose(np.asarray(d), [-2.0, 0.0])  # the periodic metric is in use
