import jax
import jax.numpy as jnp
import numpy as np

from jax_morph.core import ad_utils as ad


def test_safe_norm_value_and_zero_gradient():
    x = jnp.array([3.0, 4.0])
    assert np.isclose(ad.safe_norm(x), 5.0)
    # gradient at the zero vector is finite (0), not NaN
    g = jax.grad(lambda v: ad.safe_norm(v))(jnp.zeros(3))
    assert np.all(np.isfinite(g)) and np.allclose(g, 0.0)


def test_straight_through_forward_is_exact_and_grad_is_soft():
    hard, soft = jnp.array(1.0), jnp.array(0.3)
    assert ad.straight_through(hard, soft) == 1.0  # bit-exact forward
    g = jax.grad(lambda s: ad.straight_through(jax.lax.stop_gradient(hard), s))(soft)
    assert np.isclose(g, 1.0)


def test_heaviside_st_forward_invariant_under_temperature():
    x = jnp.array([-1.0, 0.5, 2.0])
    hard = jnp.array([0.0, 1.0, 1.0])
    for t in (0.01, 1.0, 100.0):  # forward independent of temperature
        assert np.allclose(ad.heaviside_st(x, t), hard)
    g = jax.grad(lambda v: ad.heaviside_st(v, 1.0).sum())(x)
    assert np.all(np.isfinite(g)) and np.all(g >= 0)  # sigmoid'>=0


def test_sample_bernoulli_st_forward_is_exact_sample(key):
    p = jnp.array([0.0, 1.0, 0.5])
    s = ad.sample_bernoulli_st(key, p)
    assert set(np.asarray(s).tolist()) <= {0.0, 1.0}
    assert s[0] == 0.0 and s[1] == 1.0
    g = jax.grad(lambda pp: ad.sample_bernoulli_st(key, pp).sum())(p)
    assert np.allclose(g, 1.0)  # identity surrogate


def test_sample_categorical_st_is_onehot_and_temperature_invariant(key):
    logits = jnp.array([0.1, 5.0, -2.0])
    for t in (0.1, 1.0, 10.0):
        s = ad.sample_categorical_st(key, logits, t)
        assert np.isclose(s.sum(), 1.0) and set(np.asarray(s).tolist()) <= {0.0, 1.0}


def test_safe_log_value_and_gradient_are_finite_at_and_below_zero():
    # the whole point of safe_log: finite value AND finite (zero) gradient at x <= 0, dodging the
    # classic double-where NaN trap. A regression that dropped the trick would NaN here.
    for x0 in (0.0, -3.0):
        v = float(ad.safe_log(jnp.array(x0)))
        g = float(jax.grad(ad.safe_log)(jnp.array(x0)))
        assert np.isfinite(v) and v < -1e29  # large-negative sentinel, not -inf/NaN
        assert np.isfinite(g) and np.isclose(g, 0.0)
    # away from the singularity it is the true log and its derivative
    assert np.isclose(float(ad.safe_log(jnp.array(2.0))), float(np.log(2.0)))
    assert np.isclose(float(jax.grad(ad.safe_log)(jnp.array(2.0))), 0.5)


def test_safe_divide_value_and_gradient_are_finite_at_zero_denominator():
    def div(a, b):
        return ad.safe_divide(a, b)

    # value 0 and finite (zero) gradient in BOTH arguments when b == 0
    assert np.isclose(float(div(jnp.array(5.0), jnp.array(0.0))), 0.0)
    ga = float(jax.grad(div, argnums=0)(jnp.array(5.0), jnp.array(0.0)))
    gb = float(jax.grad(div, argnums=1)(jnp.array(5.0), jnp.array(0.0)))
    assert np.all(np.isfinite([ga, gb])) and np.allclose([ga, gb], 0.0)
    # away from b == 0 it is ordinary division with the correct partials (d/da = 1/b, d/db = -a/b^2)
    assert np.isclose(float(div(jnp.array(3.0), jnp.array(4.0))), 0.75)
    assert np.isclose(float(jax.grad(div, argnums=0)(jnp.array(3.0), jnp.array(4.0))), 0.25)
    assert np.isclose(float(jax.grad(div, argnums=1)(jnp.array(3.0), jnp.array(4.0))), -3.0 / 16.0)
