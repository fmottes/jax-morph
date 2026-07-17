"""Mechanical relaxation: converged FIRE with implicit-diff equilibrium sensitivities.

``relax_equilibrium`` runs FIRE to a genuine force tolerance and differentiates the *equilibrium*
(not the solver path) by the implicit function theorem on a projected physical subspace.
``MechanicalRelaxation`` wraps it as a quasistatic step that consumes any ``Potential``.
"""

import warnings
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import cg

from ...core.state import POSITION
from ...core.step import SimulationStep, StepType
from .potentials import NoForce, Potential

__all__ = ['relax_equilibrium', 'MechanicalRelaxation']


def _fire_to_tol(
    potential,
    state,
    max_steps,
    f_tol,
    *,
    dt_start=0.01,
    dt_max=0.1,
    n_min=5,
    f_inc=1.1,
    f_dec=0.5,
    alpha_start=0.1,
    f_alpha=0.99,
):
    """Forward-only FIRE to a force tolerance. Gradients come from the implicit VJP below.

    On non-convergence (``max_steps`` reached with ``|grad U|_inf > f_tol``) the returned positions
    are not at equilibrium and the implicit-diff gradient may be inaccurate; a ``RuntimeWarning`` is
    emitted at runtime via ``jax.debug.callback`` so it surfaces even from inside a jitted call.
    """
    grad_fn = jax.grad(lambda x: potential.total_energy(x, state))

    def body(c):
        x, v, f, dt, alpha, n_pos, i = c
        uphill = jnp.vdot(f, v) <= 0.0
        n_pos = jnp.where(uphill, 0, n_pos + 1)
        can_inc = (~uphill) & (n_pos > n_min)
        dt = jnp.where(uphill, dt * f_dec, jnp.where(can_inc, jnp.minimum(dt * f_inc, dt_max), dt))
        alpha = jnp.where(uphill, alpha_start, jnp.where(can_inc, alpha * f_alpha, alpha))
        v = jnp.where(uphill, jnp.zeros_like(v), v) + dt * f
        fnorm = jnp.sqrt(jnp.sum(f * f)) + 1e-30
        v = (1.0 - alpha) * v + alpha * (f / fnorm) * jnp.sqrt(jnp.sum(v * v))
        x = x + dt * v
        return (x, v, -grad_fn(x), dt, alpha, n_pos, i + 1)

    def cond(c):
        f, i = c[2], c[6]
        return (i < max_steps) & (jnp.max(jnp.abs(f)) > f_tol)

    def _warn_not_converged(residual):
        # Re-check the tolerance host-side: under vmap the guarding lax.cond runs both branches, so the
        # callback also fires for converged lanes - only warn for a lane that genuinely missed f_tol.
        if float(residual) <= f_tol:
            return
        warnings.warn(
            f'FIRE relaxation did not reach equilibrium: the final force |grad U|_inf = '
            f'{float(residual):.2e} still exceeds f_tol = {f_tol:.2e} after max_steps = {max_steps} '
            f'steps. The returned positions are not at equilibrium, so the implicit-diff gradient may '
            f'be inaccurate; increase max_steps or raise f_tol.',
            RuntimeWarning,
            stacklevel=2,
        )

    # Carry the force through the loop so the O(N^2) gradient is evaluated once per FIRE step: body
    # computes it at the new position for the next iteration, and cond tests that carried force.
    # (Recomputing the gradient inside cond would double the force evaluations per step.)
    x0 = state.position
    init = (x0, jnp.zeros_like(x0), -grad_fn(x0), dt_start, alpha_start, 0, 0)
    final = jax.lax.while_loop(cond, body, init)
    residual = jnp.max(jnp.abs(final[2]))  # inf-norm of the force at the returned configuration
    # A runtime (not trace-time) warning: jax.debug.callback fires host-side even from inside a
    # jitted / differentiated call, and lax.cond keeps it off the converged (common) path.
    jax.lax.cond(
        residual > f_tol,
        lambda: jax.debug.callback(_warn_not_converged, residual),
        lambda: None,
    )
    return final[0]


def _rigid_body_modes(x_star, alive):
    """Orthonormal basis of the rigid-body null modes of a free-space equilibrium, on the alive cells.

    A distance-only potential is invariant under a global translation (``d`` modes) and a global
    rotation (``d(d-1)/2`` modes) of the cluster, so both are EXACT zero modes of the Hessian at the
    equilibrium. The modes are built on the alive cells about their centroid and Gram-Schmidt
    orthonormalized; a degenerate rotation (collinear cells, or a lone cell) comes out as a zero
    vector and contributes nothing. Dead-cell DOFs are handled separately by masking in ``_project``.
    """
    a = alive.astype(x_star.dtype)[:, None]  # (N, 1)
    n, d = x_star.shape
    n_alive = jnp.maximum(jnp.sum(a), 1.0)
    centered = (x_star - jnp.sum(x_star * a, axis=0, keepdims=True) / n_alive) * a
    modes = [jnp.zeros((n, d)).at[:, c].set(1.0) * a for c in range(d)]  # translations
    for p in range(d):  # rotations: one infinitesimal generator per coordinate plane
        for q in range(p + 1, d):
            w = jnp.zeros((n, d)).at[:, p].set(-centered[:, q]).at[:, q].set(centered[:, p])
            modes.append(w * a)
    basis = []  # Gram-Schmidt with safe normalization (a degenerate mode becomes exactly zero)
    for m in modes:
        for e in basis:
            m = m - jnp.sum(m * e) * e
        sq = jnp.sum(
            m * m
        )  # guard the sqrt argument so a degenerate mode (sq=0) keeps a finite grad
        keep = sq > 1e-20
        basis.append(jnp.where(keep, m / jnp.sqrt(jnp.where(keep, sq, 1.0)), 0.0))
    return basis


def _project(v, alive, basis):
    """Remove dead-cell DOFs and the rigid-body ``basis`` modes from a position cotangent.

    Zeroes dead cells (their positions do not enter the energy) and subtracts the component of ``v``
    along each orthonormal rigid-body mode, leaving only the physical (deformation) subspace on which
    the adjoint solve is well-posed. A zero mode in ``basis`` (a degenerate rotation) is a no-op.
    """
    v = v * alive.astype(v.dtype)[:, None]
    for e in basis:
        v = v - jnp.sum(v * e) * e
    return v


def _grad_x(potential, state, x):
    """Position gradient ``grad_x U`` at fixed params/state (its root is the equilibrium x*)."""
    return jax.grad(lambda xx: potential.total_energy(xx, state))(x)


# The differentiable argument is the single pytree ``params = (potential, state)`` (its float leaves
# are traced, its bool/callable leaves are not); only the hashable Python scalars
# ``max_steps``/``f_tol``/``ridge`` are nondiff. This keeps ``jax.custom_vjp`` valid under
# ``eqx.filter_jit``: ``nondiff_argnums`` must be hashable, which a state carrying array leaves (e.g.
# the ``alive`` mask) is not - so the state rides in the differentiable pytree, not there.
@partial(jax.custom_vjp, nondiff_argnums=(1, 2, 3))
def _relax_impl(params, max_steps, f_tol, ridge):
    potential, state = params
    return _fire_to_tol(potential, state, max_steps, f_tol)


def _relax_fwd(params, max_steps, f_tol, ridge):
    potential, state = params
    x_star = _fire_to_tol(potential, state, max_steps, f_tol)
    return x_star, (params, x_star)


def _relax_bwd(max_steps, f_tol, ridge, res, g):
    params, x_star = res
    _, state = params
    alive = state.alive

    # Adjoint via the implicit function theorem, restricted to the PHYSICAL subspace: project the
    # exact rigid-body null modes (dead-cell DOFs + global translation + global rotation of the alive
    # cells) out of the iterates, so a tiny ``ridge`` (for residual near-zero soft modes only) does
    # not bias the physical-mode sensitivities. Solve (P H P + ridge) u = P g on that subspace.
    basis = _rigid_body_modes(x_star, alive)  # translations + rotations, built once and reused
    project = lambda v: _project(v, alive, basis)
    hess = lambda v: jax.jvp(lambda x: _grad_x(*params, x), (x_star,), (v,))[1]

    def op(v):
        vp = project(v)
        return project(hess(vp)) + ridge * vp

    u, _ = cg(op, project(g), maxiter=500)
    u = project(u)

    # dL/dparams = -u^T d(grad_x U)/dparams = the vjp of grad_x U wrt params applied to -u. Since
    # params is (potential, state), this returns cotangents for BOTH the potential's parameters and
    # the incoming state (radii -> sigma, ...); non-float leaves (alive, callables) get float0.
    _, vjp_params = jax.vjp(lambda p: _grad_x(*p, x_star), params)
    (g_params,) = vjp_params(-u)
    return (g_params,)


_relax_impl.defvjp(_relax_fwd, _relax_bwd)


def relax_equilibrium(potential, state, *, max_steps=500, f_tol=1e-3, ridge=1e-6):
    """Relax positions to mechanical equilibrium, with implicit-diff (not solver-path) gradients.

    The forward runs FIRE to a genuine force tolerance ``|grad U| <= f_tol`` (a real equilibrium, not
    a fixed step count). The backward is the implicit-function-theorem sensitivity of that
    equilibrium, restricted to the physical subspace: the rigid-body null modes (dead-cell DOFs, a
    global translation, and a global rotation of the alive cells) are projected out, and a tiny
    ``ridge`` regularizes only residual near-zero soft modes. Differentiates w.r.t. both the
    potential's parameters and the incoming ``state`` (radii -> sigma, ...).

    If FIRE hits ``max_steps`` before reaching ``f_tol`` it emits a runtime ``RuntimeWarning`` and
    returns the last iterate anyway (behaviour is otherwise unchanged); raise ``max_steps`` or
    ``f_tol`` to silence it.

    Pass ``potential=None`` for ``NoForce``, whose every configuration is already an equilibrium, so
    the positions are returned unchanged (matching ``MechanicalRelaxation``).

    Args:
        potential: Interaction potential, or None for `NoForce`.
        state: Input state whose positions provide the FIRE initial condition.
        max_steps: Static maximum number of FIRE iterations. Defaults to 500.
        f_tol: Maximum absolute force tolerated at convergence. Defaults to 1e-3.
        ridge: Regularization for residual soft modes in the implicit adjoint. Defaults to 1e-6.

    Returns:
        Relaxed position array with shape ``(capacity, n_space_dim)``.
    """
    if potential is None:
        potential = NoForce()
    return _relax_impl((potential, state), max_steps, f_tol, ridge)


class MechanicalRelaxation(SimulationStep):
    r"""Quasistatic step: relax positions to mechanical equilibrium under ``potential`` each step.

    Each step drives positions to a force balance $\nabla_x U = 0$ with FIRE and differentiates that
    equilibrium by the implicit function theorem, not the solver path.

    Gradient behaviour (easy to misread): the equilibrium sensitivity is taken on the *physical*
    (deformation) subspace only. The rigid-body null modes of a free-space equilibrium - the dead
    cells' degrees of freedom, a global translation, and a global rotation of the alive cells - are
    projected out of the adjoint and carry **no gradient at all**. This is deliberate: the energy is
    flat along those gauge modes (the relaxed cluster's absolute position and orientation are fixed by
    the initial condition, not the parameters). A shape objective is invariant to them and unaffected;
    a translation- or rotation-sensitive objective simply gets no gradient for that sensitivity
    (rather than a spurious ridge-scaled one). To make the cluster's position or orientation
    optimizable, break the symmetry physically (pin a cell, add an external field / substrate).

    Attributes:
        potential: Interaction potential defining the equilibrium.
        max_steps: Static maximum number of FIRE iterations. Defaults to 500.
        f_tol: Maximum absolute force tolerated at convergence. Defaults to 1e-3.
        ridge: Regularization for residual soft modes in the implicit adjoint. Defaults to 1e-6.
    """

    step_type = StepType.QUASISTATIC
    potential: Potential  # any Potential (its total_energy defines the equilibrium)
    max_steps: int = eqx.field(static=True)  # loop bound -> static
    f_tol: float  # plain numeric fields
    ridge: float

    def __init__(self, potential, *, max_steps=500, f_tol=1e-3, ridge=1e-6):
        """Build the step from a potential and the FIRE / adjoint tolerances.

        Pass ``potential=None`` for ``NoForce``, whose every configuration is already an equilibrium,
        so relaxation is a no-op (positions pass through unchanged).

        Args:
            potential: Interaction potential, or None for `NoForce`.
            max_steps: Static maximum number of FIRE iterations. Defaults to 500.
            f_tol: Maximum absolute force tolerated at convergence. Defaults to 1e-3.
            ridge: Regularization for residual soft modes in the implicit adjoint. Defaults to
                1e-6.
        """
        self.potential = potential if potential is not None else NoForce()
        self.max_steps = max_steps
        self.f_tol = f_tol
        self.ridge = ridge

    def state_reads(self):
        """Reads positions (the FIRE start), plus any state field the potential sources params from."""
        return (POSITION, *self.potential.state_reads())

    def state_writes(self):
        """Writes the relaxed positions."""
        return (POSITION,)

    def __call__(self, state, *, dt, key):
        """Return state with positions relaxed to mechanical equilibrium.

        Args:
            state: Input state with positions of shape ``(capacity, n_space_dim)``.
            dt: Unused macro-step duration.
            key: Unused JAX PRNG key.

        Returns:
            State with relaxed positions of shape ``(capacity, n_space_dim)``.
        """
        x = relax_equilibrium(
            self.potential, state, max_steps=self.max_steps, f_tol=self.f_tol, ridge=self.ridge
        )
        return state.set('position', x)
