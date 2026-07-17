"""Screened diffusion: the dimension-correct finite-size analytic Green's function."""

import jax.numpy as jnp
import numpy as np
import pytest

from jax_morph.core.state import build_state_from_model
from jax_morph.physics.diffusion import FreeScreenedDiffusion


def _seed(diff, *, capacity, n_space_dim, n_types=1):
    # Build a state whose schema carries the step's read (secretion_rate) and write (chemical)
    # fields; a step's state_requires() is the union of its reads and writes.
    return build_state_from_model(diff).init_empty(
        capacity=capacity, n_space_dim=n_space_dim, n_types=n_types
    )


def test_3d_finite_sphere_kernel_and_self_term():
    diff = FreeScreenedDiffusion(n_field_species=1, n_space_dim=3, diffusion=2.0, degradation=0.5)
    s = _seed(diff, capacity=2, n_space_dim=3)
    s = s.update(
        alive=s.alive.at[:2].set(True),
        radius=s.radius.at[:2].set(0.2),
        position=jnp.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]),
        secretion_rate=jnp.array([[1.0], [0.0]]),  # only cell 0 secretes
    )
    chem = diff(s, dt=1.0, key=None)['chemical']
    D, k, a = 2.0, 0.5, 0.2
    kappa = np.sqrt(k / D)
    r = 1.5
    expected = np.exp(-kappa * (r - a)) / (4 * np.pi * D * r * (1 + kappa * a))  # finite sphere
    assert np.isclose(float(chem[1, 0]), expected, rtol=1e-6)
    self_expected = 1.0 / (4 * np.pi * D * a * (1 + kappa * a))  # r_eff=a on-surface value
    assert np.isclose(float(chem[0, 0]), self_expected, rtol=1e-6)


def test_2d_finite_disk_kernel_vs_scipy():
    from scipy.special import k0, k1  # independent reference for the A&S K0/K1 approximations

    diff = FreeScreenedDiffusion(n_field_species=1, n_space_dim=2, diffusion=1.0, degradation=1.0)
    s = _seed(diff, capacity=2, n_space_dim=2)
    s = s.update(
        alive=s.alive.at[:2].set(True),
        radius=s.radius.at[:2].set(0.2),
        position=jnp.array([[0.0, 0.0], [1.5, 0.0]]),
        secretion_rate=jnp.array([[1.0], [0.0]]),
    )
    chem = diff(s, dt=1.0, key=None)['chemical']
    D, kk, a = 1.0, 1.0, 0.2
    kappa = np.sqrt(kk / D)
    expected = k0(kappa * 1.5) / (2 * np.pi * D * a * kappa * k1(kappa * a))  # finite disk
    assert np.isclose(float(chem[1, 0]), expected, rtol=1e-5)  # validates _k0 and _k1 vs scipy
    self_expected = k0(kappa * a) / (2 * np.pi * D * a * kappa * k1(kappa * a))  # r_eff=a self term
    assert np.isclose(float(chem[0, 0]), self_expected, rtol=1e-5)  # 2-D near/self clamp


def test_dead_cells_produce_no_nans():
    # Dead cells have radius 0; the finite-size self/near kernel must stay NaN-free. They are
    # masked out, but an unfloored 1/r_eff (3-D) or K1(0) (2-D) singularity would give inf*0 = NaN
    # and trip jax_debug_nans. Capacity > n_seed exercises the dead-source path.
    diff = FreeScreenedDiffusion(n_field_species=1, n_space_dim=3, diffusion=2.0, degradation=0.5)
    s = _seed(diff, capacity=5, n_space_dim=3)
    s = s.update(
        alive=s.alive.at[:2].set(True),
        radius=s.radius.at[:2].set(0.2),
        position=jnp.array(
            [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        ),
        secretion_rate=jnp.ones((5, 1)),  # even dead cells carry a source value
    )
    chem = diff(s, dt=1.0, key=None)['chemical']
    assert np.all(np.isfinite(np.asarray(chem)))
    assert np.allclose(np.asarray(chem)[2:], 0.0)  # dead receivers masked to zero


def test_dead_cells_produce_no_nans_2d():
    # The 2-D headline hazard: an unfloored K1(0) at a dead source (radius 0) would give inf*0 = NaN
    # under debug_nans. This drives the exact singular branch the a-floor exists to guard.
    diff = FreeScreenedDiffusion(n_field_species=1, n_space_dim=2, diffusion=1.0, degradation=1.0)
    s = _seed(diff, capacity=5, n_space_dim=2)
    s = s.update(
        alive=s.alive.at[:2].set(True),
        radius=s.radius.at[:2].set(0.2),
        position=jnp.array([[0.0, 0.0], [1.5, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
        secretion_rate=jnp.ones((5, 1)),  # even dead cells carry a source value
    )
    chem = diff(s, dt=1.0, key=None)['chemical']
    assert np.all(np.isfinite(np.asarray(chem)))
    assert np.allclose(np.asarray(chem)[2:], 0.0)  # dead receivers masked to zero


def test_traced_zero_degradation_is_nan_free_2d():
    # A traced degradation that hits 0 during optimization (a static 0 is rejected at construction)
    # would give kappa=0 -> K1(0) NaN in 2-D; the kappa floor returns a large finite value instead.
    diff = FreeScreenedDiffusion(
        n_field_species=1, n_space_dim=2, diffusion=1.0, degradation=jnp.array(0.0)
    )
    s = _seed(diff, capacity=2, n_space_dim=2)
    s = s.update(
        alive=s.alive.at[:2].set(True),
        radius=s.radius.at[:2].set(0.2),
        position=jnp.array([[0.0, 0.0], [1.5, 0.0]]),
        secretion_rate=jnp.array([[1.0], [0.0]]),
    )
    assert np.all(np.isfinite(np.asarray(diff(s, dt=1.0, key=None)['chemical'])))


def test_traced_zero_degradation_is_nan_free_1d():
    # 1-D analogue: kappa=0 gives 1/(2 D kappa) = inf without the floor; the floor keeps it finite.
    diff = FreeScreenedDiffusion(
        n_field_species=1, n_space_dim=1, diffusion=1.0, degradation=jnp.array(0.0)
    )
    s = _seed(diff, capacity=2, n_space_dim=1)
    s = s.update(
        alive=s.alive.at[:2].set(True),
        radius=s.radius.at[:2].set(0.2),
        position=jnp.array([[0.0], [1.5]]),
        secretion_rate=jnp.array([[1.0], [0.0]]),
    )
    assert np.all(np.isfinite(np.asarray(diff(s, dt=1.0, key=None)['chemical'])))


def test_multi_species_uses_per_species_coefficients():
    # Two field species with distinct (diffusion, degradation) and distinct source rates: each
    # species must be solved with its OWN coefficients (guards the vmap in_axes/out_axes wiring).
    D, K = jnp.array([2.0, 3.0]), jnp.array([0.5, 1.0])
    diff = FreeScreenedDiffusion(n_field_species=2, n_space_dim=3, diffusion=D, degradation=K)
    s = _seed(diff, capacity=2, n_space_dim=3, n_types=1)
    s = s.update(
        alive=s.alive.at[:2].set(True),
        radius=s.radius.at[:2].set(0.2),
        position=jnp.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]),
        secretion_rate=jnp.array([[1.0, 2.0], [0.0, 0.0]]),  # cell 0 secretes both species
    )
    chem = np.asarray(diff(s, dt=1.0, key=None)['chemical'])
    a, r = 0.2, 1.5
    for sp, (Dc, Kc, S0) in enumerate([(2.0, 0.5, 1.0), (3.0, 1.0, 2.0)]):
        kappa = np.sqrt(Kc / Dc)
        cross = np.exp(-kappa * (r - a)) / (4 * np.pi * Dc * r * (1 + kappa * a))
        assert np.isclose(chem[1, sp], S0 * cross, rtol=1e-6)  # per-species cross term
        self_term = 1.0 / (4 * np.pi * Dc * a * (1 + kappa * a))
        assert np.isclose(chem[0, sp], S0 * self_term, rtol=1e-6)  # per-species self term


def test_n_space_dim_must_match_state():
    # n_space_dim selects the disk (2-D) vs sphere (3-D) kernel; a mismatch with the state's actual
    # spatial dimension would silently apply the wrong-dimension kernel, so it is rejected up front.
    diff = FreeScreenedDiffusion(n_field_species=1, n_space_dim=3)  # built for 3-D
    s = _seed(diff, capacity=2, n_space_dim=2)  # but the state is 2-D
    s = s.update(
        alive=s.alive.at[:2].set(True),
        radius=s.radius.at[:2].set(0.2),
        position=jnp.array([[0.0, 0.0], [1.0, 0.0]]),
        secretion_rate=jnp.ones((2, 1)),
    )
    with pytest.raises(ValueError, match='n_space_dim'):
        diff(s, dt=1.0, key=None)


def test_invalid_n_space_dim_rejected():
    # Only 1-D (segment), 2-D (disk), and 3-D (sphere) kernels exist; a stray n_space_dim must not
    # fall through to the 2-D branch silently.
    with pytest.raises(ValueError, match='n_space_dim must be'):
        FreeScreenedDiffusion(n_field_species=1, n_space_dim=4)


def test_low_dimension_needs_positive_degradation():
    # The unscreened steady field is unbounded in 1-D and 2-D (kappa -> 0 diverges); a statically zero
    # degradation is caught at construction rather than surfacing as a NaN under debug_nans. 3-D is fine.
    for d in (1, 2):
        with pytest.raises(ValueError, match='degradation > 0'):
            FreeScreenedDiffusion(n_field_species=1, n_space_dim=d, degradation=0.0)
    FreeScreenedDiffusion(
        n_field_species=1, n_space_dim=3, degradation=0.0
    )  # 3-D unscreened is bounded


def test_1d_finite_segment_kernel_and_self_term():
    # 1-D finite-segment screened Green's function: G(r) = exp(-kappa (r_eff - a)) / (2 D kappa),
    # with r_eff = max(r, a). Validate a cross term and the on-surface self term.
    diff = FreeScreenedDiffusion(n_field_species=1, n_space_dim=1, diffusion=2.0, degradation=0.5)
    s = _seed(diff, capacity=2, n_space_dim=1)
    s = s.update(
        alive=s.alive.at[:2].set(True),
        radius=s.radius.at[:2].set(0.2),
        position=jnp.array([[0.0], [1.5]]),
        secretion_rate=jnp.array([[1.0], [0.0]]),  # only cell 0 secretes
    )
    chem = diff(s, dt=1.0, key=None)['chemical']
    D, k, a, r = 2.0, 0.5, 0.2, 1.5
    kappa = np.sqrt(k / D)
    assert np.isclose(float(chem[1, 0]), np.exp(-kappa * (r - a)) / (2 * D * kappa), rtol=1e-6)
    assert np.isclose(float(chem[0, 0]), 1.0 / (2 * D * kappa), rtol=1e-6)  # r_eff = a self term


def test_custom_source_and_field_names():
    # The read/write field names are constructor arguments, so the same solver models any diffusing
    # quantity (here 'heat_source' -> 'temperature'); the physics is identical to the default naming.
    diff = FreeScreenedDiffusion(
        n_field_species=1,
        n_space_dim=3,
        diffusion=2.0,
        degradation=0.5,
        source_name='heat_source',
        field_name='temperature',
    )
    assert diff.state_reads()[0].name == 'heat_source'
    assert diff.state_writes()[0].name == 'temperature'
    s = _seed(diff, capacity=2, n_space_dim=3)
    s = s.update(
        alive=s.alive.at[:2].set(True),
        radius=s.radius.at[:2].set(0.2),
        position=jnp.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]),
        heat_source=jnp.array([[1.0], [0.0]]),
    )
    out = diff(s, dt=1.0, key=None)
    D, k, a, r = 2.0, 0.5, 0.2, 1.5
    kappa = np.sqrt(k / D)
    expected = np.exp(-kappa * (r - a)) / (4 * np.pi * D * r * (1 + kappa * a))
    assert np.isclose(float(out['temperature'][1, 0]), expected, rtol=1e-6)
