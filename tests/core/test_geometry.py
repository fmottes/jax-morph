import jax.numpy as jnp
import numpy as np

from jax_morph.core import geometry as geo


def test_free_space_displacement_and_shift():
    disp, shift = geo.free_space()
    a, b = jnp.array([1.0, 2.0]), jnp.array([0.0, 0.0])
    assert np.allclose(disp(a, b), [1.0, 2.0])
    assert np.allclose(shift(a, jnp.array([1.0, 1.0])), [2.0, 3.0])


def test_pairwise_displacements():
    disp, _ = geo.free_space()
    pos = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]])
    d = geo.pairwise_displacements(pos, disp)
    assert d.shape == (3, 3, 2)
    assert np.allclose(d[0, 1], [-1.0, 0.0])  # pos[0] - pos[1]
    assert np.allclose(d[2, 0], [0.0, 2.0])
    assert np.allclose(d[0, 0], 0.0) and np.allclose(d[1, 1], 0.0)  # self displacement zero


def test_periodic_minimum_image():
    disp, shift = geo.periodic(box=10.0)
    a, b = jnp.array([9.0, 0.0]), jnp.array([1.0, 0.0])
    assert np.allclose(disp(a, b), [-2.0, 0.0])  # wraps: 8 -> -2 (minimum image)
    assert np.allclose(shift(jnp.array([9.0, 0.0]), jnp.array([2.0, 0.0])), [1.0, 0.0])


def test_neighbor_sum_masks_self_and_dead():
    alive = jnp.array([True, True, False])
    vals = jnp.ones((3, 3, 2))  # per-pair vector contributions
    out = geo.neighbor_sum(vals, alive)
    assert out.shape == (3, 2)
    assert np.allclose(out[0], [1.0, 1.0])  # cell 0 sums only over j=1 (self & dead excluded)
    assert np.allclose(out[2], [0.0, 0.0])  # dead cell contributes nothing
