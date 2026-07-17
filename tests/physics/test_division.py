"""Division: volume-conserving, oriented, non-overlapping daughters with lineage recording."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from jax_morph.core.simulate import simulate, transition_logp
from jax_morph.core.state import StateFieldSpec, build_state_from_model
from jax_morph.core.step import Model, SimulationStep, StepType, check_stochastic_step
from jax_morph.physics.division import Division, reconstruct_lineage


class _CarryMemory(SimulationStep):
    """No-op step that only puts a persistent, non-heritable per-cell field into the schema."""

    step_type = StepType.QUASISTATIC

    def state_writes(self):
        return (StateFieldSpec('memory', heritable=False, default=-9.0),)

    def __call__(self, state, *, dt, key):
        return state


def _divide_state(capacity, n_seed, *, n_space_dim=2):
    # A state for the Division step alone: the generated schema carries division_rate, division_axis,
    # division_overflow, born, mother, and the ephemeral trace fields. Seed n_seed live cells.
    s = build_state_from_model(Division(n_space_dim=n_space_dim)).init_empty(
        capacity=capacity, n_space_dim=n_space_dim, n_types=1
    )
    return s.update(
        alive=s.alive.at[:n_seed].set(True),
        radius=s.radius.at[:n_seed].set(0.5),
        celltype=s.celltype.at[:n_seed, 0].set(1.0),
    )


def test_division_creates_volume_conserving_touching_daughter(key):
    s = _divide_state(capacity=4, n_seed=1)
    s = s.set('division_rate', s.division_rate.at[0].set(50.0))  # large hazard -> p ~ 1
    s2 = Division(n_space_dim=2)(s, dt=1.0, key=key)  # p = 1 - exp(-50) ~ 1
    assert int(s2.alive.sum()) == 2
    idx = np.where(np.asarray(s2.alive))[0]
    m = 2.0 ** (-1.0 / 2)
    assert np.allclose(np.asarray(s2.radius)[idx], 0.5 * m, atol=1e-6)  # volume-conserving radius
    d = float(jnp.linalg.norm(s2.position[idx[0]] - s2.position[idx[1]]))
    assert np.isclose(d, 2 * 0.5 * m, atol=1e-5)  # centers 2r' apart (touching)


def test_division_is_dt_consistent(key):
    # p = 1 - exp(-lambda*dt): the fraction of cells dividing must scale as 1 - exp(-lambda*dt).
    lam, big = 0.5, 4096
    s = _divide_state(capacity=2 * big, n_seed=big)
    s = s.set('division_rate', jnp.full((2 * big,), lam))
    for dt in (0.1, 0.2):
        s_dt = Division(n_space_dim=2)(s, dt=dt, key=key)
        frac = (int(s_dt.alive.sum()) - big) / big
        assert np.isclose(frac, 1.0 - np.exp(-lam * dt), atol=0.03)  # dt-consistent hazard


def test_zero_division_rate_is_exactly_off(key):
    step = Division(n_space_dim=2)
    s = _divide_state(capacity=8, n_seed=4)
    s = s.set('division_rate', jnp.zeros((8,)))

    assert np.array_equal(np.asarray(step._dist(s, 1.0)), np.zeros((8,)))
    out = step(s, dt=1.0, key=key)
    trace = step.trace_from_state(out)
    assert np.array_equal(np.asarray(trace['divided']), np.zeros((8,)))
    assert float(step.logp(s, trace, 1.0)) == 0.0


def test_division_overflow_is_capped_and_recorded(key):
    # no free slots (capacity == n_seed) and both cells want to divide -> both dropped, recorded
    s = _divide_state(capacity=2, n_seed=2)
    s = s.set('division_rate', jnp.full((2,), 50.0))
    s2 = Division(n_space_dim=2)(s, dt=1.0, key=key)
    assert int(s2.alive.sum()) == 2  # capacity-capped: no new cells created
    assert float(s2.division_overflow) == 2.0  # both dropped divisions recorded, no crash


def test_division_placement_is_oriented(key):
    # orientation_snr >> 1 with a fixed axis: the daughter is offset along +/- axis (|cos| ~ 1).
    s = _divide_state(capacity=4, n_seed=1)
    s = s.update(
        division_rate=s.division_rate.at[0].set(50.0),
        division_axis=s.division_axis.at[0].set(jnp.array([1.0, 0.0])),
    )
    s2 = Division(n_space_dim=2, orientation_snr=50.0)(s, dt=1.0, key=key)
    idx = np.where(np.asarray(s2.alive))[0]
    sep = np.asarray(s2.position[idx[0]] - s2.position[idx[1]])
    sep = sep / np.linalg.norm(sep)
    assert abs(sep[0]) > 0.99  # aligned with the x axis


def test_division_records_born_and_mother(key):
    s = _divide_state(capacity=4, n_seed=1)
    s = s.set('division_rate', s.division_rate.at[0].set(50.0))
    s2 = Division(n_space_dim=2)(s, dt=1.0, key=key)
    born, mother = np.asarray(s2.born), np.asarray(s2.mother)
    assert int(born.sum()) == 1  # exactly one new daughter
    daughter = int(np.where(born > 0.5)[0][0])
    assert mother[daughter] == 0  # its parent is slot 0
    assert born[0] == 0.0 and mother[0] == -1  # the mother is not "born"; carries the sentinel


def test_division_trace_round_trips(key):
    # replay must co-emit divided / divide_eligible / division_dir so scoring reads the trace back.
    s = _divide_state(capacity=4, n_seed=2)
    s = s.set('division_rate', jnp.full((4,), 5.0))
    check_stochastic_step(Division(n_space_dim=2), s, dt=1.0, key=key)


def test_logp_scores_only_eligible_cells(key):
    # Division.logp(state, trace, dt) sums bernoulli_logp over the alive-at-decision cells only.
    step = Division(n_space_dim=2)
    s = _divide_state(capacity=6, n_seed=3)  # 3 eligible, 3 dead
    s = s.set('division_rate', jnp.full((6,), 0.4))
    s2 = step(s, dt=1.0, key=key)
    trace = step.trace_from_state(s2)
    p = 1.0 - np.exp(-0.4)
    divided, eligible = np.asarray(trace['divided']), np.asarray(trace['divide_eligible'])
    ref = np.sum((divided * np.log(p) + (1 - divided) * np.log(1 - p)) * eligible)
    assert np.isclose(float(step.logp(s, trace, 1.0)), ref, atol=1e-5)
    assert np.allclose(eligible[3:], 0.0)  # dead cells are not eligible -> not scored


def test_pathwise_gradient_reaches_the_hazard_rate(key):
    # grad(sum radii) wrt division_rate flows through the backward-only Bernoulli surrogate: a
    # committed cell shrinks (radius *= 2^(-1/d)), and raising its rate raises that shrink, so the
    # gradient is nonzero for cells that divided. A moderate rate keeps dp/dlambda from saturating,
    # and many seeded cells guarantee several dividers under the fixed key.
    step = Division(n_space_dim=2)
    s = _divide_state(capacity=32, n_seed=16)

    def loss(rate):
        return jnp.sum(step(s.set('division_rate', rate), dt=1.0, key=key).radius)

    g = eqx.filter_grad(loss)(jnp.full((32,), 1.0))
    # raising a cell's rate raises its shrink, so the gradient of summed radii is non-positive
    # everywhere and strictly negative for cells that divided (non-dividers/daughters contribute 0).
    assert float(jnp.max(g)) <= 1e-9
    assert float(jnp.min(g)) < -1e-6


def test_reconstruct_lineage_from_history():
    # slot 0 (founder) divides into slot 1 at step 1, then slot 0 divides into slot 2 at step 2.
    alive = np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1]], dtype=bool)
    born = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    mother = np.array([[-1, -1, -1], [-1, 0, -1], [-1, -1, 0]], dtype=int)
    nodes = reconstruct_lineage(born, mother, alive)
    founders = [n for n in nodes if n['parent'] is None]
    daughters = [n for n in nodes if n['parent'] is not None]
    assert len(founders) == 1 and len(daughters) == 2  # one founder + two daughters
    assert all(dnode['parent'] == founders[0]['id'] for dnode in daughters)
    assert {dnode['birth_step'] for dnode in daughters} == {1, 2}


def test_reconstruct_lineage_ignores_stale_birth_trace_on_chained_initial_frame(key):
    model = Model([Division(n_space_dim=2)])
    s0 = _divide_state(capacity=4, n_seed=1)
    s0 = s0.set('division_rate', s0.division_rate.at[0].set(50.0))
    first = simulate(model, s0, n_steps=1, key=key, history=True)
    chained_initial = jax.tree_util.tree_map(lambda x: x[-1], first)
    assert int(chained_initial.born.sum()) == 1

    chained_initial = chained_initial.set(
        'division_rate', jnp.zeros_like(chained_initial.division_rate)
    )
    second = simulate(model, chained_initial, n_steps=1, key=key, history=True)
    nodes = reconstruct_lineage(second.born, second.mother, second.alive)

    assert len(nodes) == 2
    assert {node['slot'] for node in nodes} == {0, 1}
    assert all(node['parent'] is None and node['birth_step'] == 0 for node in nodes)


def test_partial_overflow_and_multi_divider_ordering(key):
    # 4 cells all want to divide but only 2 free slots: exactly 2 commit (cumsum rank order), 2 drop.
    s = _divide_state(capacity=6, n_seed=4)
    s = s.set('division_rate', jnp.full((6,), 50.0))
    s2 = Division(n_space_dim=2)(s, dt=1.0, key=key)
    assert int(s2.alive.sum()) == 6  # 4 mothers + 2 committed daughters
    assert float(s2.division_overflow) == 2.0  # 2 dividers dropped
    born, mother = np.asarray(s2.born), np.asarray(s2.mother)
    assert int(born.sum()) == 2  # exactly 2 new daughters
    daughters = np.where(born > 0.5)[0]
    assert set(daughters) == {4, 5}  # the two free slots, in order
    assert sorted(int(mother[d]) for d in daughters) == [0, 1]  # lowest-index dividers commit first
    m = 2.0 ** (-1.0 / 2)
    for dtr in daughters:  # every committed daughter is volume-conserving and touching its mother
        mo = int(mother[dtr])
        assert np.isclose(float(s2.radius[dtr]), 0.5 * m, atol=1e-6)
        sep = float(jnp.linalg.norm(s2.position[dtr] - s2.position[mo]))
        assert np.isclose(sep, 2 * 0.5 * m, atol=1e-5)


def test_replay_is_deterministic_given_the_recorded_trace(key):
    # division_dir is recorded, so replaying the recorded trace reproduces the exact daughters the
    # forward produced (the property transition replay relies on). A re-drawn direction would fail.
    step = Division(n_space_dim=2)
    s = _divide_state(capacity=6, n_seed=3)
    s = s.set('division_rate', jnp.full((6,), 5.0))
    s1 = step(s, dt=1.0, key=key)
    trace = step.trace_from_state(s1)
    r_a = step.replay(s, trace, dt=1.0, pathwise=False)
    r_b = step.replay(s, trace, dt=1.0, pathwise=False)
    assert np.allclose(
        np.asarray(r_a.position), np.asarray(r_b.position)
    )  # replay is deterministic
    assert np.allclose(np.asarray(r_a.position), np.asarray(s1.position))  # and matches the forward
    assert np.array_equal(np.asarray(r_a.alive), np.asarray(s1.alive))


def test_transition_logp_is_finite_and_deterministic(key):
    model = Model([Division(n_space_dim=2)])
    s = _divide_state(capacity=6, n_seed=3)
    s = s.set('division_rate', jnp.full((6,), 1.0))
    s_next = model(s, dt=1.0, key=key)
    lp1 = float(transition_logp(model, s, s_next, dt=1.0))
    lp2 = float(transition_logp(model, s, s_next, dt=1.0))
    assert np.isfinite(lp1) and lp1 == lp2


def test_isotropic_and_all_dead_placement_are_nan_free(key):
    # zero division_axis + default orientation_snr -> isotropic direction via safe_norm; and an
    # all-dead state must produce no divisions and no NaN under jax_debug_nans.
    step = Division(n_space_dim=2)  # orientation_snr=0, division_axis default 0
    s = _divide_state(capacity=4, n_seed=2)
    s = s.set('division_rate', jnp.full((4,), 1.0))
    out = step(s, dt=1.0, key=key)
    assert np.all(np.isfinite(np.asarray(out.position)))
    s0 = build_state_from_model(Division(n_space_dim=2)).init_empty(
        capacity=4, n_space_dim=2, n_types=1
    )
    s0 = s0.set('division_rate', jnp.full((4,), 1.0))
    out0 = step(s0, dt=1.0, key=key)
    assert int(out0.alive.sum()) == 0
    assert np.all(np.isfinite(np.asarray(out0.position)))


def test_daughter_resets_non_heritable_field_to_default(key):
    # A daughter takes a (possibly recycled) free slot; a non-heritable field there must be reset to
    # its default, not inherit the slot's stale value. Slot 1 is dead but holds a stale 777.0.
    model = Model([_CarryMemory(), Division(n_space_dim=2)])
    s = build_state_from_model(model).init_empty(capacity=4, n_space_dim=2, n_types=1)
    s = s.update(
        alive=s.alive.at[0].set(True),
        radius=s.radius.at[0].set(0.5),
        division_rate=s.division_rate.at[0].set(50.0),
        # mother carries a non-default non-heritable value; the free slot holds a stale one
        memory=s.memory.at[0].set(42.0).at[1].set(777.0),
    )
    s2 = Division(n_space_dim=2)(s, dt=1.0, key=key)  # cell 0 divides into slot 1
    assert bool(s2.alive[1])  # slot 1 is now the daughter
    assert (
        float(s2.memory[1]) == -9.0
    )  # daughter reset to default (not the stale 777, not inherited)
    assert float(s2.memory[0]) == 42.0  # the mother's own non-heritable value is preserved
