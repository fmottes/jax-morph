"""Numerically-safe ops and forward-exact / backward-smooth straight-through primitives.

Design invariant: every straight-through function's forward output is
bit-identical to the exact operation and independent of the surrogate ``temperature``;
only the backward pass uses the smooth surrogate.
"""

import jax
import jax.numpy as jnp

__all__ = [
    'safe_norm',
    'safe_log',
    'safe_divide',
    'straight_through',
    'heaviside_st',
    'argmax_st',
    'sample_categorical_st',
    'sample_bernoulli_st',
    'bernoulli_logp',
    'categorical_logp',
    'gaussian_logp',
]


def safe_norm(x, axis=-1, keepdims=False):
    """Compute a Euclidean norm with value and gradient zero at the zero vector.

    Args:
        x: Input array.
        axis: Axis or axes over which to compute the norm. Defaults to -1.
        keepdims: Whether reduced axes remain with size one. Defaults to False.

    Returns:
        The norm of ``x`` with the requested reduced shape.
    """
    sq = jnp.sum(jnp.square(x), axis=axis, keepdims=keepdims)
    safe_sq = jnp.where(sq == 0.0, 1.0, sq)
    return jnp.where(sq == 0.0, 0.0, jnp.sqrt(safe_sq))


def safe_log(x):
    """Compute a log that stays finite with a finite gradient for nonpositive inputs.

    Args:
        x: Input values.

    Returns:
        ``log(x)`` where ``x > 0``, and ``-1e30`` otherwise.
    """
    safe_x = jnp.where(x > 0.0, x, 1.0)
    return jnp.where(x > 0.0, jnp.log(safe_x), -1e30)


def safe_divide(a, b):
    """Divide two arrays with zero value and gradient where the denominator is zero.

    Args:
        a: Numerator values.
        b: Denominator values, broadcastable with ``a``.

    Returns:
        ``a / b`` where ``b`` is nonzero, and zero otherwise.
    """
    safe_b = jnp.where(b == 0.0, 1.0, b)
    return jnp.where(b == 0.0, 0.0, a / safe_b)


def straight_through(hard, soft):
    """Return ``hard`` exactly while differentiating through ``soft``.

    Args:
        hard: Values to use in the forward pass.
        soft: Shape-compatible surrogate values for the backward pass.

    Returns:
        An array equal to ``hard`` whose gradient is that of ``soft``.
    """
    return hard + (soft - jax.lax.stop_gradient(soft))


def heaviside_st(x, temperature=1.0):
    """Apply an exact Heaviside threshold with a sigmoid surrogate gradient.

    Args:
        x: Values to threshold at zero.
        temperature: Positive sigmoid temperature for the backward pass. Defaults to 1.0.

    Returns:
        Exact zero-or-one threshold values with sigmoid surrogate gradients.
    """
    hard = jnp.where(x > 0.0, 1.0, 0.0)
    return straight_through(hard, jax.nn.sigmoid(x / temperature))


def argmax_st(logits, temperature=1.0, axis=-1):
    """Select an exact one-hot argmax with a softmax surrogate gradient.

    Args:
        logits: Unnormalized category logits.
        temperature: Positive softmax temperature for the backward pass. Defaults to 1.0.
        axis: Category axis. Defaults to -1.

    Returns:
        One-hot argmax values with softmax surrogate gradients.
    """
    hard = jax.nn.one_hot(jnp.argmax(logits, axis=axis), logits.shape[axis], axis=axis)
    return straight_through(hard, jax.nn.softmax(logits / temperature, axis=axis))


def sample_categorical_st(key, logits, temperature=1.0, axis=-1):
    """Draw an exact one-hot categorical sample with a softmax surrogate gradient.

    Args:
        key: JAX PRNG key.
        logits: Unnormalized category logits.
        temperature: Positive softmax temperature for the backward pass. Defaults to 1.0.
        axis: Category axis. Defaults to -1.

    Returns:
        One-hot categorical samples with softmax surrogate gradients.
    """
    idx = jax.random.categorical(key, logits, axis=axis)
    hard = jax.nn.one_hot(idx, logits.shape[axis], axis=axis)
    return straight_through(hard, jax.nn.softmax(logits / temperature, axis=axis))


def sample_bernoulli_st(key, p):
    """Draw an exact Bernoulli sample with an identity surrogate gradient.

    Args:
        key: JAX PRNG key.
        p: Success probabilities.

    Returns:
        Zero-or-one samples whose derivative with respect to ``p`` is one.
    """
    hard = jax.random.bernoulli(key, p).astype(p.dtype)
    return straight_through(hard, p)


def bernoulli_logp(outcome, p):
    """Elementwise Bernoulli log-density ``log p(outcome | p)`` for ``outcome`` in ``{0, 1}``.

    Pairs with ``sample_bernoulli_st``: recompute ``p`` from the state (through the policy params)
    and score the recorded 0/1 outcome. Uses ``safe_log`` so ``p`` at 0 or 1 gives a finite value
    and gradient rather than NaN.

    Args:
        outcome: Recorded zero-or-one outcomes.
        p: Success probabilities, broadcastable with ``outcome``.

    Returns:
        Elementwise Bernoulli log-probabilities.
    """
    return outcome * safe_log(p) + (1.0 - outcome) * safe_log(1.0 - p)


def categorical_logp(onehot, logits, axis=-1):
    """Categorical log-density: ``log softmax(logits)`` selected by the one-hot ``outcome``.

    Pairs with ``sample_categorical_st``: recompute ``logits`` from the state (through the policy
    params) and score the recorded one-hot action. Reduces over ``axis`` (the class axis).

    Args:
        onehot: Recorded one-hot outcomes.
        logits: Unnormalized category logits.
        axis: Category axis to reduce. Defaults to -1.

    Returns:
        Log-probabilities with the category axis removed.
    """
    return jnp.sum(onehot * jax.nn.log_softmax(logits, axis=axis), axis=axis)


def gaussian_logp(x, mean, std):
    """Elementwise Gaussian log-density ``log N(x | mean, std**2)`` (``std`` is the deviation).

    Pairs with a reparameterized draw ``x = mean + std * xi``: recompute ``mean``/``std`` from the
    state (through the policy params) and score the recorded value ``x`` - the density a Brownian /
    Langevin step assigns to its displacement. ``mean`` and ``std`` broadcast against ``x``; ``std``
    is assumed strictly positive (a real noise scale), so no boundary guard is applied.

    Args:
        x: Recorded values.
        mean: Gaussian means, broadcastable with ``x``.
        std: Strictly positive standard deviations, broadcastable with ``x``.

    Returns:
        Elementwise Gaussian log-densities.
    """
    z = (x - mean) / std
    return -0.5 * z * z - jnp.log(std) - 0.5 * jnp.log(2.0 * jnp.pi)
