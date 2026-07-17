"""Space (boundary conditions) and dense pairwise geometry.

The space (``displacement`` / ``shift``) is injectable at state construction - ``free_space()``
(default) or ``periodic(box)`` - and physics steps read ``state.displacement``. Whether a given
physics module is valid in a given space is the user's responsibility; the library does not police
modeling choices (e.g. the free-space ``FreeScreenedDiffusion`` paired with ``periodic`` is a
modeling error, documented on the step, not enforced). ``neighbor_sum`` is a dense N x N reduction
combinator for custom pairwise layers; a sparse/neighbor-list backend would be a future migration of
it and its callers, not a runtime flag.
"""

import jax
import jax.numpy as jnp

__all__ = ['free_space', 'periodic', 'pairwise_displacements', 'neighbor_sum']


def _free_displacement(Ra, Rb):
    return Ra - Rb


def _free_shift(R, dR):
    return R + dR


def free_space():
    """Free-space (unbounded) metric: displacement = Ra - Rb, shift = R + dR.

    Returns the same module-level singleton functions on every call. The state stores the space as
    static fields (``displacement``/``shift``), so sharing one identity keeps every free-space state
    the same PyTree structure - a jitted ``simulate``/``logp`` compiles once, not once per initial
    state.

    Returns:
        A ``(displacement, shift)`` pair implementing an unbounded space.
    """
    return _free_displacement, _free_shift


_PERIODIC_CACHE: dict = {}


def periodic(box):
    """Periodic orthorhombic box of side lengths ``box`` (scalar or (dim,)); minimum-image.

    Memoised on the box value so ``periodic(box)`` returns the same function objects for equal
    boxes - equal-box states then share one static space identity and do not force a recompile
    (same discipline as ``free_space`` and the generated state-class cache).

    Args:
        box: Scalar side length or vector of per-axis side lengths.

    Returns:
        A memoized ``(displacement, shift)`` pair implementing minimum-image wrapping.
    """
    box = jnp.asarray(box, float)
    key = float(box) if box.ndim == 0 else tuple(float(v) for v in box.reshape(-1))
    cached = _PERIODIC_CACHE.get(key)
    if cached is not None:
        return cached

    def displacement(Ra, Rb):
        d = Ra - Rb
        return d - box * jnp.round(d / box)

    def shift(R, dR):
        return jnp.mod(R + dR, box)

    _PERIODIC_CACHE[key] = (displacement, shift)
    return _PERIODIC_CACHE[key]


def _callable_descriptor(function):
    """Return a diagnostic callable identity when it is stable enough to report."""
    module = getattr(function, '__module__', None)
    qualname = getattr(function, '__qualname__', None)
    if not isinstance(module, str) or not isinstance(qualname, str):
        return None
    if '<locals>' in qualname or '<lambda>' in qualname:
        return None
    return f'{module}:{qualname}'


def _space_descriptor(space):
    """Describe a built-in space by callable identity, or a custom space diagnostically."""
    try:
        displacement, shift = space
    except (TypeError, ValueError) as error:
        raise TypeError('space must be a (displacement, shift) pair') from error
    if displacement is _free_displacement and shift is _free_shift:
        return {'kind': 'free'}
    for key, cached_space in _PERIODIC_CACHE.items():
        if (displacement is cached_space[0]) and (shift is cached_space[1]):
            box = key if isinstance(key, float) else list(key)
            box_array = jnp.asarray(box, float)
            return {
                'kind': 'periodic',
                'box': box,
                'box_dtype': box_array.dtype.str,
            }
    return {
        'kind': 'custom',
        'displacement': _callable_descriptor(displacement),
        'shift': _callable_descriptor(shift),
    }


def _space_from_descriptor(descriptor, *, space=None):
    """Recreate a saved built-in space or validate a caller-supplied replacement space."""
    if not isinstance(descriptor, dict) or not isinstance(descriptor.get('kind'), str):
        raise ValueError('space descriptor must be an object with a kind')
    kind = descriptor['kind']
    if kind == 'free':
        expected = free_space()
    elif kind == 'periodic':
        box = descriptor.get('box')
        if not isinstance(box, (int, float, list)) or isinstance(box, bool):
            raise ValueError('periodic space descriptor has an invalid box')
        if isinstance(box, list) and (
            not box
            or any(not isinstance(value, (int, float)) or isinstance(value, bool) for value in box)
        ):
            raise ValueError('periodic space descriptor has an invalid box')
        box_dtype = descriptor.get('box_dtype')
        if not isinstance(box_dtype, str):
            raise ValueError('periodic space descriptor has an invalid box dtype')
        try:
            saved_dtype = jnp.dtype(box_dtype)
        except TypeError as error:
            raise ValueError('periodic space descriptor has an invalid box dtype') from error
        if not jnp.issubdtype(saved_dtype, jnp.floating):
            raise ValueError('periodic space descriptor box dtype must be floating')
        current_dtype = jnp.asarray(box, float).dtype
        if current_dtype != saved_dtype:
            raise ValueError(
                f'periodic box dtype {saved_dtype.str!r} cannot be reconstructed exactly; '
                'enable the required JAX dtype support'
            )
        expected = periodic(box)
    elif kind == 'custom':
        if space is None:
            raise ValueError('custom saved space requires an explicit space=(displacement, shift)')
        supplied = _space_descriptor(space)
        for name in ('displacement', 'shift'):
            saved_name = descriptor.get(name)
            if saved_name is not None and supplied.get(name) != saved_name:
                raise ValueError(f'custom supplied space has a different {name} callable')
        return space
    else:
        raise ValueError(f'unsupported space descriptor kind {kind!r}')

    if space is None:
        return expected
    if _space_descriptor(space) != descriptor:
        raise ValueError('supplied space does not match the saved built-in space descriptor')
    return space


def pairwise_displacements(positions, displacement):
    """(N, N, dim) array with ``[i, j] = displacement(positions[i], positions[j])``.

    The fundamental pairwise geometry primitive; distances are ``safe_norm(disp, axis=-1)`` at the
    call site (some steps want only displacements, some only distances).

    Args:
        positions: Cell positions of shape ``(N, dim)``.
        displacement: Binary displacement function supplied by the state's space.

    Returns:
        Pairwise displacement vectors of shape ``(N, N, dim)``.
    """
    return jax.vmap(lambda ri: jax.vmap(lambda rj: displacement(ri, rj))(positions))(positions)


def neighbor_sum(pair_values, alive):
    """Reduce a per-pair tensor over sources: ``sum_j pair_values[i, j]`` over alive ``j != i``.

    ``pair_values`` has shape ``(N, N, *rest)`` and the result is ``(N, *rest)``. A **dense
    reduction combinator** that keeps custom pairwise layers short with uniform alive/self masking.
    It consumes a materialized dense N x N tensor, so a sparse/neighbor-list backend replaces both
    it AND its callers - a real future migration, not a drop-in.

    Args:
        pair_values: Per-pair values of shape ``(N, N, *rest)``.
        alive: Boolean liveness mask of shape ``(N,)``.

    Returns:
        Per-cell sums of shape ``(N, *rest)``, excluding self and dead-cell pairs.
    """
    a = alive.astype(pair_values.dtype)
    n = a.shape[0]
    m = jnp.outer(a, a) * (1.0 - jnp.eye(n, dtype=pair_values.dtype))
    m = m.reshape(m.shape + (1,) * (pair_values.ndim - 2))
    return jnp.sum(pair_values * m, axis=1)
