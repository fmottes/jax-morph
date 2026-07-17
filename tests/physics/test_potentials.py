"""Tests for the interaction-potential zoo and the shared pair machinery.

The whole suite runs with ``jax_debug_nans`` on (see ``tests/conftest.py``), so a NaN produced for a
masked pair - e.g. a dead-dead padded pair with ``sigma = 0`` in the sigma-relative cutoff - raises
rather than being silently masked. ``_padded_state`` builds exactly that padded configuration.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jax_morph as jxm
from jax_morph.core.state import build_state_from_model
from jax_morph.physics.mechanics import (
    BrownianDynamics,
    Harmonic,
    Hertzian,
    LennardJones,
    MechanicalRelaxation,
    Morse,
    SoftSphere,
)


class _M:  # bare model: no declared fields, so the state carries only the base fields
    def state_requires(self):
        return ()


def _padded_state(positions, *, radius=0.5, capacity=None, n_types=1):
    """A state with the given live cells followed by padded dead cells (origin, radius 0).

    ``capacity > len(positions)`` leaves dead-dead pairs whose contact distance ``sigma = r_i + r_j``
    is 0, which exercises the sigma-relative cutoff's zero-denominator guard.
    """
    positions = jnp.asarray(positions, dtype=float)
    n, dim = positions.shape
    capacity = n if capacity is None else capacity
    s = build_state_from_model(_M()).init_empty(capacity=capacity, n_space_dim=dim, n_types=n_types)
    return s.update(
        alive=s.alive.at[:n].set(True),
        radius=s.radius.at[:n].set(radius),
        position=s.position.at[:n].set(positions),
        celltype=s.celltype.at[:n, 0].set(1.0),
    )


def _two(dist, radius=0.5):
    """A two-cell state at separation ``dist`` (contact distance sigma = 2 * radius)."""
    return _padded_state(jnp.array([[0.0, 0.0], [dist, 0.0]]), radius=radius)


REPULSIVE = [SoftSphere, Hertzian]


def test_morse_padded_dead_cells_do_not_nan():
    # Dead padded cells sit at the origin with radius 0, so a dead-dead pair has sigma = 0 and the
    # cutoff denominator (r_off - r_on) is 0: an unguarded 0/0 makes total_energy NaN and, under the
    # suite's debug_nans, makes forces/virial raise. The safe_divide guard keeps them finite.
    pot = Morse()
    s = _padded_state(jnp.array([[0.0, 0.0], [1.0, 0.0]]), capacity=4)  # 2 alive, 2 dead
    e = pot.total_energy(s.position, s)
    assert np.isfinite(float(e))
    assert np.all(np.isfinite(np.asarray(pot.forces(s))))
    assert np.all(np.isfinite(np.asarray(pot.virial_pressure(s))))
    # the dead pairs contribute nothing: the padded energy equals the unpadded two-cell energy
    s_bare = _padded_state(jnp.array([[0.0, 0.0], [1.0, 0.0]]))  # capacity == n_alive
    assert np.isclose(float(e), float(pot.total_energy(s_bare.position, s_bare)))


def test_morse_grad_through_padded_state_is_finite():
    # jax.grad of a padded-state energy w.r.t. a traced epsilon must not NaN through the masked pairs.
    s = _padded_state(jnp.array([[0.0, 0.0], [1.0, 0.0]]), capacity=4)
    g = jax.grad(lambda e: Morse(epsilon=e).total_energy(s.position, s))(jnp.array(3.0))
    assert np.isfinite(float(g))


# ---------------------------------------------------------------------------
# SoftSphere and Hertzian: the compact purely-repulsive family.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('Pot', REPULSIVE)
def test_compact_repulsion_zero_at_and_beyond_contact(Pot):
    pot = Pot(epsilon=2.0)
    at = _two(1.0)  # sigma = 1.0 -> exactly at contact
    beyond = _two(1.2)  # r > sigma -> switched fully off
    assert np.isclose(float(pot.total_energy(at.position, at)), 0.0)
    assert np.allclose(np.asarray(pot.forces(at)), 0.0, atol=1e-7)
    assert np.isclose(float(pot.total_energy(beyond.position, beyond)), 0.0)
    assert np.allclose(np.asarray(pot.forces(beyond)), 0.0)


@pytest.mark.parametrize('Pot', REPULSIVE)
def test_compact_repulsion_positive_under_compression(Pot):
    pot = Pot(epsilon=2.0)
    s = _two(0.6)  # r < sigma -> repulsive
    assert float(pot.total_energy(s.position, s)) > 0.0
    assert np.all(np.asarray(pot.virial_pressure(s)) > 0.0)  # compression -> positive pressure


@pytest.mark.parametrize('Pot', REPULSIVE)
def test_compact_repulsion_forces_match_autodiff(Pot):
    pot = Pot(epsilon=2.0)
    s = _two(0.7)
    g = jax.grad(lambda p: pot.total_energy(p, s))(s.position)
    assert np.allclose(np.asarray(pot.forces(s)), np.asarray(-g))


@pytest.mark.parametrize('Pot', REPULSIVE)
def test_compact_repulsion_padded_nan_safe(Pot):
    # padded state mixes compressed pairs, a beyond-cutoff pair, and dead-dead pairs (sigma = 0).
    pot = Pot(epsilon=2.0)
    s = _padded_state(jnp.array([[0.0, 0.0], [0.6, 0.0], [1.3, 0.0]]), capacity=6)
    assert np.isfinite(float(pot.total_energy(s.position, s)))
    assert np.all(np.isfinite(np.asarray(pot.forces(s))))
    assert np.all(np.isfinite(np.asarray(pot.virial_pressure(s))))
    g = jax.grad(lambda e: Pot(epsilon=e).total_energy(s.position, s))(jnp.array(2.0))
    assert np.isfinite(float(g))


@pytest.mark.parametrize('Pot', REPULSIVE)
def test_compact_repulsion_per_cell_epsilon(Pot):
    # A StateFieldSpec coupling is read per cell and combined per pair by the arithmetic mean.
    spec = jxm.StateFieldSpec('stiffness')
    pot = Pot(epsilon=spec)
    assert pot.state_reads() == (spec,)
    model = jxm.Model([MechanicalRelaxation(pot, max_steps=1, f_tol=1e-6)])
    s = build_state_from_model(model).init_empty(capacity=2, n_space_dim=2, n_types=1)
    s = s.update(
        alive=s.alive.at[:2].set(True),
        radius=s.radius.at[:2].set(0.5),
        position=jnp.array([[0.0, 0.0], [0.6, 0.0]]),
        stiffness=s.stiffness.at[:2].set(jnp.array([2.0, 6.0])),  # per-cell -> mean 4.0
    )
    ref = _two(0.6)
    assert np.isclose(
        float(pot.total_energy(s.position, s)),
        float(Pot(epsilon=4.0).total_energy(ref.position, ref)),
    )


def test_softsphere_energy_matches_closed_form():
    # U = (eps / 2) (1 - r / sigma)^2 for r < sigma.
    s = _two(0.6)  # sigma = 1.0, r = 0.6
    assert np.isclose(
        float(SoftSphere(epsilon=2.0).total_energy(s.position, s)),
        0.5 * 2.0 * (1.0 - 0.6) ** 2,
    )


def test_hertzian_softer_onset_than_softsphere():
    # just inside contact, the Hertzian force ~(1 - r/sigma)^1.5 is smaller than SoftSphere's ~(1 - r/sigma).
    s = _two(0.95)
    f_soft = float(np.linalg.norm(np.asarray(SoftSphere(epsilon=2.0).forces(s))[0]))
    f_hertz = float(np.linalg.norm(np.asarray(Hertzian(epsilon=2.0).forces(s))[0]))
    assert 0.0 < f_hertz < f_soft


# ---------------------------------------------------------------------------
# Harmonic: the standard spring shifted so it decays to 0 at (and beyond) the cutoff.
# ---------------------------------------------------------------------------


def test_harmonic_negative_well_minimum_at_contact():
    # shifted so U -> 0 at the cutoff, giving a negative well of depth (k/2)(r_cut - sigma)^2 at contact.
    pot = Harmonic(k=3.0)
    s = _two(1.0)  # r = sigma = 1.0 -> well minimum
    depth = 0.5 * 3.0 * (2.5 - 1.0) ** 2  # (k/2)(r_cutoff_frac * sigma - sigma)^2
    assert np.isclose(float(pot.total_energy(s.position, s)), -depth)
    assert np.allclose(np.asarray(pot.forces(s)), 0.0, atol=1e-6)  # zero force at the minimum


def test_harmonic_clean_linear_force_no_wiggle():
    # the restoring force is the undistorted linear spring inside the cutoff: attractive (cell 0
    # pulled toward cell 1, +x) at every stretched separation, with magnitude k * (r - sigma).
    pot = Harmonic(k=3.0)
    for d in [1.2, 1.6, 2.0, 2.4]:
        fx = float(np.asarray(pot.forces(_two(d)))[0, 0])
        assert np.isclose(fx, 3.0 * (d - 1.0))  # exactly the linear spring, no wiggle or bend


def test_harmonic_restoring_force_sign():
    # cell 0 at the origin, cell 1 at +x: compressed -> cell 0 pushed to -x; stretched -> pulled to +x.
    pot = Harmonic(k=3.0)
    fx_compressed = float(np.asarray(pot.forces(_two(0.6)))[0, 0])  # r < sigma
    fx_stretched = float(np.asarray(pot.forces(_two(1.3)))[0, 0])  # sigma < r < cutoff
    assert fx_compressed < 0.0 < fx_stretched


def test_harmonic_zero_energy_and_gradient_beyond_cutoff():
    # the shifted well decays to 0 at the cutoff, so a non-interacting pair contributes nothing to the
    # energy OR its radius-gradient (not merely zero force) - no spurious offset from far-apart pairs.
    pot = Harmonic(k=3.0)
    for d in [2.6, 10.0]:  # both beyond the 2.5 cutoff
        s = _two(d)
        assert np.isclose(
            float(pot.total_energy(s.position, s)), 0.0
        )  # non-interacting -> 0 energy
        assert np.allclose(np.asarray(pot.forces(s)), 0.0)  # and zero force
    # d(total_energy)/d(radius) is 0 for a non-interacting pair (the clamp had a spurious 4.5 here)
    energy = lambda radius: Harmonic(k=1.0).total_energy(
        _two(10.0, radius).position, _two(10.0, radius)
    )
    assert np.isclose(float(jax.grad(energy)(0.5)), 0.0)


def test_harmonic_forces_match_autodiff():
    pot = Harmonic(k=3.0)
    s = _two(0.7)
    g = jax.grad(lambda p: pot.total_energy(p, s))(s.position)
    assert np.allclose(np.asarray(pot.forces(s)), np.asarray(-g))


def test_harmonic_virial_sign_compression_and_tension():
    pot = Harmonic(k=3.0)
    assert np.all(np.asarray(pot.virial_pressure(_two(0.6))) > 0.0)  # compression -> positive
    assert np.all(np.asarray(pot.virial_pressure(_two(1.3))) < 0.0)  # tension -> negative


def test_harmonic_padded_nan_safe():
    pot = Harmonic(k=3.0)
    s = _padded_state(jnp.array([[0.0, 0.0], [0.6, 0.0], [1.3, 0.0]]), capacity=6)
    assert np.isfinite(float(pot.total_energy(s.position, s)))
    assert np.all(np.isfinite(np.asarray(pot.forces(s))))
    assert np.all(np.isfinite(np.asarray(pot.virial_pressure(s))))


def test_harmonic_per_cell_stiffness_from_state():
    spec = jxm.StateFieldSpec('stiffness')
    pot = Harmonic(k=spec)
    assert pot.state_reads() == (spec,)
    model = jxm.Model([MechanicalRelaxation(pot, max_steps=1, f_tol=1e-6)])
    s = build_state_from_model(model).init_empty(capacity=2, n_space_dim=2, n_types=1)
    s = s.update(
        alive=s.alive.at[:2].set(True),
        radius=s.radius.at[:2].set(0.5),
        position=jnp.array([[0.0, 0.0], [0.6, 0.0]]),
        stiffness=s.stiffness.at[:2].set(jnp.array([2.0, 6.0])),  # per-cell -> mean k = 4.0
    )
    ref = _two(0.6)
    assert np.isclose(
        float(pot.total_energy(s.position, s)),
        float(Harmonic(k=4.0).total_energy(ref.position, ref)),
    )


def test_harmonic_relaxes_overlapping_pair_to_contact():
    pot = Harmonic(k=5.0)
    relax = MechanicalRelaxation(pot, max_steps=800, f_tol=1e-8)
    s2 = relax(_two(0.5), dt=1.0, key=None)  # overlapping -> spring pulls to the r = sigma minimum
    d = float(jnp.linalg.norm(s2.position[0] - s2.position[1]))
    assert abs(d - 1.0) < 0.02


# ---------------------------------------------------------------------------
# LennardJones: hard r^-12 core + r^-6 adhesive tail, minimum -epsilon at contact.
# ---------------------------------------------------------------------------


def test_lennardjones_minimum_at_contact():
    pot = LennardJones(epsilon=2.0)
    s = _two(1.0)  # r = sigma -> the well minimum, energy -epsilon per pair
    assert np.isclose(float(pot.total_energy(s.position, s)), -2.0)
    assert np.allclose(np.asarray(pot.forces(s)), 0.0, atol=1e-6)


def test_lennardjones_repulsive_core_and_attractive_tail():
    pot = LennardJones(epsilon=2.0)
    # the r_min form crosses zero at 2^(-1/6) sigma ~ 0.89 sigma, so the energy is positive only
    # deeper in the core; test a clearly-repulsive point and a clearly-attractive one.
    assert float(pot.total_energy(_two(0.8).position, _two(0.8))) > 0.0  # hard core
    assert float(pot.total_energy(_two(1.2).position, _two(1.2))) < 0.0  # adhesive tail


def test_lennardjones_compact_beyond_cutoff():
    pot = LennardJones(epsilon=2.0)
    s = _two(3.0)  # r > r_cutoff_frac * sigma = 2.5 -> switched fully off
    assert np.isclose(float(pot.total_energy(s.position, s)), 0.0)
    assert np.allclose(np.asarray(pot.forces(s)), 0.0)


def test_lennardjones_forces_match_autodiff():
    pot = LennardJones(epsilon=2.0)
    s = _two(0.95)
    g = jax.grad(lambda p: pot.total_energy(p, s))(s.position)
    assert np.allclose(np.asarray(pot.forces(s)), np.asarray(-g))


def test_lennardjones_virial_positive_under_compression():
    pot = LennardJones(epsilon=2.0)
    assert np.all(np.asarray(pot.virial_pressure(_two(0.85))) > 0.0)  # r < sigma -> compression


def test_lennardjones_padded_nan_safe():
    # Exercises both r=0 hazards: the self-diagonal (sigma > 0) and dead-dead pairs (sigma = 0),
    # where a naive (sigma/r)^12 would be inf. safe_divide keeps sigma/r finite there and the masked
    # pairs are zeroed by neighbor_sum, so energy, forces, and virial stay finite under debug_nans.
    pot = LennardJones(epsilon=2.0)
    s = _padded_state(jnp.array([[0.0, 0.0], [0.95, 0.0], [1.2, 0.0]]), capacity=6)
    assert np.isfinite(float(pot.total_energy(s.position, s)))
    assert np.all(np.isfinite(np.asarray(pot.forces(s))))
    assert np.all(np.isfinite(np.asarray(pot.virial_pressure(s))))
    g = jax.grad(lambda e: LennardJones(epsilon=e).total_energy(s.position, s))(jnp.array(2.0))
    assert np.isfinite(float(g))


def test_lennardjones_per_cell_epsilon_from_state():
    spec = jxm.StateFieldSpec('adhesion')
    pot = LennardJones(epsilon=spec)
    assert pot.state_reads() == (spec,)
    model = jxm.Model([MechanicalRelaxation(pot, max_steps=1, f_tol=1e-6)])
    s = build_state_from_model(model).init_empty(capacity=2, n_space_dim=2, n_types=1)
    s = s.update(
        alive=s.alive.at[:2].set(True),
        radius=s.radius.at[:2].set(0.5),
        position=jnp.array([[0.0, 0.0], [1.0, 0.0]]),  # at contact -> -epsilon_ij = -(2 + 6) / 2
        adhesion=s.adhesion.at[:2].set(jnp.array([2.0, 6.0])),
    )
    assert np.isclose(float(pot.total_energy(s.position, s)), -4.0)


def test_lennardjones_implicit_diff_separation_sensitivity():
    # The two-cell equilibrium separation is sigma = 2 * radius, so d(separation)/d(radius) = 2;
    # implicit diff through the FIRE solve must recover it at the LJ well.
    relax = MechanicalRelaxation(LennardJones(epsilon=2.0), max_steps=1500, f_tol=1e-9)

    def separation(radius):
        s = _two(1.0, radius=radius)
        s2 = relax(s, dt=1.0, key=None)
        return jnp.linalg.norm(s2.position[0] - s2.position[1])

    assert np.isclose(float(jax.grad(separation)(0.5)), 2.0, atol=0.05)


# ---------------------------------------------------------------------------
# Drop-in integration: the whole zoo works with the dynamics steps unchanged (the
# "for free" property). Repulsive vs adhesive behave distinctly.
# ---------------------------------------------------------------------------


ADHESIVE = [Harmonic(k=3.0), LennardJones(epsilon=2.0)]
NEW_POTENTIALS = [SoftSphere(epsilon=2.0), Hertzian(epsilon=2.0), *ADHESIVE]


def _brownian_state(bd, dist=0.9, radius=0.5):
    s = build_state_from_model(bd).init_empty(capacity=2, n_space_dim=2, n_types=1)
    return s.update(
        alive=s.alive.at[:2].set(True),
        radius=s.radius.at[:2].set(radius),
        position=jnp.array([[0.0, 0.0], [dist, 0.0]]),
    )


@pytest.mark.parametrize('pot', NEW_POTENTIALS, ids=lambda p: type(p).__name__)
def test_relaxation_separates_overlapping_pair(pot):
    # every new potential is a drop-in MechanicalRelaxation potential: an overlapping pair separates
    # to (at least) the contact distance - the well minimum for the adhesive ones, the point where
    # the repulsive force vanishes for SoftSphere / Hertzian.
    relax = MechanicalRelaxation(pot, max_steps=1500, f_tol=1e-6)
    s0 = _two(
        0.8
    )  # overlapping (sigma = 1.0); a gentle overlap so LJ's stiff core does not launch it
    s2 = relax(s0, dt=1.0, key=None)
    d0 = float(jnp.linalg.norm(s0.position[0] - s0.position[1]))
    d1 = float(jnp.linalg.norm(s2.position[0] - s2.position[1]))
    assert d1 > d0 and 0.98 <= d1 <= 1.2  # pushed out to about the contact distance
    assert float(jnp.max(jnp.abs(pot.forces(s2)))) < 1e-4  # genuinely relaxed


@pytest.mark.parametrize('pot', NEW_POTENTIALS, ids=lambda p: type(p).__name__)
def test_brownian_step_valid_for_each_potential(pot, key):
    # every new potential is a drop-in BrownianDynamics potential: one step returns a finite sparse
    # delta of the right shape, reproducible under a fixed key.
    bd = BrownianDynamics(pot, n_space_dim=2, gamma=1.0, kT=0.05)
    s = _brownian_state(bd, dist=0.9)
    d1 = bd(s, dt=0.01, key=key)
    d2 = bd(s, dt=0.01, key=key)
    assert d1.position.shape == (2, 2) and np.all(np.isfinite(np.asarray(d1.position)))
    assert np.allclose(np.asarray(d1.position), np.asarray(d2.position))  # reproducible under a key


def test_adhesive_brownian_condenses_repulsive_does_not():
    # A cold Brownian rollout of a compact cloud: the adhesive Harmonic spring holds it together (mean
    # pairwise distance shrinks) while the purely repulsive SoftSphere pushes it apart. Harmonic is the
    # adhesive exemplar because the hard cores of LJ / Morse are numerically stiff for explicit
    # Euler-Maruyama and blow up when cold noise drives a pair into the core.
    n = 12
    start = 2.0 * jax.random.uniform(jax.random.PRNGKey(3), (n, 2))  # a compact cluster
    iu = np.triu_indices(n, 1)
    start_mean = float(
        np.linalg.norm(np.asarray(start)[:, None] - np.asarray(start)[None], axis=-1)[iu].mean()
    )

    def mean_pair_distance(pot):
        model = jxm.Model([BrownianDynamics(pot, n_space_dim=2, gamma=1.0, kT=0.03)])
        s0 = build_state_from_model(model).init_empty(capacity=n, n_space_dim=2, n_types=1)
        s0 = s0.update(
            alive=s0.alive.at[:].set(True),
            radius=s0.radius.at[:].set(0.5),
            position=s0.position.at[:].set(start),
        )
        sT = jxm.simulate(model, s0, n_steps=300, dt=0.02, key=jax.random.PRNGKey(8))
        p = np.asarray(sT.position)
        return float(np.linalg.norm(p[:, None] - p[None], axis=-1)[iu].mean())

    harmonic_mean = mean_pair_distance(Harmonic(k=6.0))
    softsphere_mean = mean_pair_distance(SoftSphere(epsilon=3.0))
    assert harmonic_mean < start_mean  # adhesion holds the cloud together
    assert softsphere_mean > start_mean  # repulsion pushes it apart
    assert harmonic_mean < softsphere_mean


# ---------------------------------------------------------------------------
# Construction-time input validation (couplings must be scalar/per-cell-spec;
# the smooth-cutoff window must be ordered).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('Pot', [Morse, SoftSphere, Hertzian, LennardJones])
def test_coupling_rejects_non_scalar_array(Pot):
    # a raw (N,) or (n_types, n_types) coupling would mis-broadcast row-wise into an asymmetric
    # matrix; reject it at construction with a clear error (per-cell values go through a StateFieldSpec).
    with pytest.raises(ValueError, match='scalar'):
        Pot(epsilon=jnp.array([1.0, 2.0, 3.0]))


def test_coupling_rejects_non_scalar_state_field_spec():
    with pytest.raises(ValueError, match='per-cell scalar'):
        Morse(epsilon=jxm.StateFieldSpec('adhesion', shape=(2,)))


def test_scalar_and_per_cell_couplings_are_accepted():
    # the three valid forms construct without error
    Morse(epsilon=3.0)
    Morse(epsilon=jnp.array(3.0))
    Morse(epsilon=jxm.StateFieldSpec('adhesion'))


@pytest.mark.parametrize('Pot', [Morse, LennardJones])
def test_cutoff_window_must_be_ordered(Pot):
    # r_onset_frac >= r_cutoff_frac silently collapses the smooth switch to a discontinuous hard step;
    # reject it at construction.
    with pytest.raises(ValueError, match='r_onset_frac < r_cutoff_frac'):
        Pot(r_onset_frac=2.5, r_cutoff_frac=2.5)
    with pytest.raises(ValueError, match='r_onset_frac < r_cutoff_frac'):
        Pot(r_onset_frac=3.0, r_cutoff_frac=2.5)


def test_harmonic_cutoff_must_be_beyond_contact():
    with pytest.raises(ValueError, match='r_cutoff_frac > 1'):
        Harmonic(r_cutoff_frac=0.9)
