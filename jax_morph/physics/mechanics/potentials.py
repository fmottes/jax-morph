"""Interaction potentials: the protocol, the pairwise base, and a library of pair potentials.

The concrete potentials are Morse, soft-sphere, Hertzian, harmonic, and Lennard-Jones. Cutoffs are
sigma-relative (``sigma = r_i + r_j`` is the contact distance) so interactions stay size-consistent
as cells grow; forces come from autodiff of the energy, with no jax-md dependency. A custom
interaction is a ``PairwisePotential`` subclass supplying ``pair_params`` and ``pair_energy``, which
inherits energy, forces, and virial pressure from the base.
"""

import equinox as eqx
import jax
import jax.numpy as jnp

from ...core.ad_utils import safe_divide, safe_norm
from ...core.geometry import neighbor_sum, pairwise_displacements
from ...core.state import StateFieldSpec

__all__ = [
    'Potential',
    'NoForce',
    'PairwisePotential',
    'Morse',
    'SoftSphere',
    'Hertzian',
    'Harmonic',
    'LennardJones',
]


def _compact_repulsion(r, sigma, eps, exponent, prefactor):
    """``prefactor * eps * (1 - r/sigma)**exponent`` for ``r < sigma``, else 0 (value and grad safe).

    ``safe_divide`` handles a dead-dead padded pair (``sigma = 0``); the double ``where`` evaluates
    the (possibly fractional) power only on a strictly positive base, so the gradient is finite for
    ``r >= sigma`` too under ``debug_nans``.
    """
    base = 1.0 - safe_divide(r, sigma)
    safe = jnp.where(base > 0.0, base, 1.0)  # the power only ever sees a positive base
    return jnp.where(base > 0.0, prefactor * eps * safe**exponent, 0.0)


def _smooth_cutoff(r, r_on, r_off):
    """Multiplicative isotropic cutoff (jax-md form): a switch from 1 (r <= r_on) to 0 (r >= r_off).

    ``S(r) = (r_off^2 - r^2)^2 (r_off^2 + 2 r^2 - 3 r_on^2) / (r_off^2 - r_on^2)^3`` on the transition
    window. It is C1 (value and first derivative continuous at both ends), so the force stays
    continuous with a small slope kink at the window edges. ``safe_divide`` guards the denominator for
    dead-dead padded pairs (``sigma = 0`` gives ``r_on = r_off = 0``), which would otherwise be
    ``0/0`` and NaN a masked pair under the always-on ``jax_debug_nans``.
    """
    r2, ron2, roff2 = r * r, r_on * r_on, r_off * r_off
    s = safe_divide((roff2 - r2) ** 2 * (roff2 + 2.0 * r2 - 3.0 * ron2), (roff2 - ron2) ** 3)
    return jnp.where(r < r_on, 1.0, jnp.where(r < r_off, s, 0.0))


class Potential(eqx.Module):
    """Interaction-potential protocol.

    Implement ``total_energy(positions, state) -> scalar``; ``forces`` follows via autodiff. Any
    custom energy (not just pairwise) can subclass this - a relaxation or a Brownian drift consumes
    only ``total_energy`` / ``forces``.

    Methods:
        total_energy: Evaluate total interaction energy at supplied positions.
        forces: Differentiate total energy into per-cell forces.
        state_reads: Declare non-base state fields read by the energy.
    """

    def total_energy(self, positions, state):
        """Evaluate total interaction energy at supplied positions.

        Args:
            positions: Cell positions of shape ``(capacity, n_space_dim)``.
            state: State supplying all non-position energy inputs.

        Returns:
            Scalar total interaction energy.

        Raises:
            NotImplementedError: Always, until a subclass supplies an energy.
        """
        raise NotImplementedError(f'{type(self).__name__} must implement total_energy')

    def forces(self, state):
        """Differentiate total energy into per-cell forces at ``state.position``.

        Args:
            state: State supplying positions and any energy parameters.

        Returns:
            Force array of shape ``(capacity, n_space_dim)``.
        """
        return -jax.grad(lambda p: self.total_energy(p, state))(state.position)

    def state_reads(self):
        """State fields this energy consumes beyond the always-present base fields (default: none).

        The wrapping steps (relaxation, Brownian, stress) merge this into their own ``state_reads`` so
        that a custom field a potential reads is allocated by ``build_state_from_model``. Override to
        declare any such field; ``PairwisePotential`` already returns its from-state couplings.

        Returns:
            Tuple of additional `StateFieldSpec` objects. Defaults to an empty tuple.
        """
        return ()


class NoForce(Potential):
    """The trivial potential: no interactions, so every cell feels zero energy and zero force.

    The neutral default for the position-moving steps (``BrownianDynamics``,
    ``ActiveBrownianDynamics2D``, ``MechanicalRelaxation``) when they are built without a potential -
    a free Brownian / active-Brownian gas, or a relaxation that leaves positions untouched. It
    overrides ``forces`` to return zeros directly, skipping the autodiff of ``total_energy`` and so
    also the dense O(N^2) pairwise sum: the right base for large-N active-matter / flocking models
    whose only coupling lives in a separate step (e.g. a neighbour-alignment torque).
    """

    def total_energy(self, positions, state):
        """Always zero (no interactions)."""
        return jnp.asarray(0.0)

    def forces(self, state):
        """Zero force on every cell (overrides the autodiff path - no energy to differentiate)."""
        return jnp.zeros_like(state.position)


class PairwisePotential(Potential):
    """A pair potential defined by ``pair_energy(r, *params)`` and per-pair ``pair_params(state)``.

    Implement the elementwise ``pair_energy`` and the per-pair ``pair_params`` (a tuple of (N, N)
    arrays); ``total_energy``, ``forces``, and per-cell virial ``virial_pressure`` come for free - the
    standard way to add a soft-matter interaction (harmonic, Lennard-Jones, Hertzian, ...).
    Reductions go through ``neighbor_sum`` (the sparse seam), which masks self-pairs and dead cells.

    Methods:
        pair_energy: Evaluate an elementwise energy for each pair separation.
        pair_params: Construct per-pair parameter matrices from a state.
        mix: Convert one per-cell coupling into a symmetric pair coupling.
        virial_pressure: Evaluate per-cell virial pressure.
    """

    def pair_energy(self, r, *params):
        """Evaluate elementwise pair energy at a separation.

        Args:
            r: Pair separation array of shape ``(capacity, capacity)``.
            *params: Per-pair parameter arrays broadcastable with ``r``.

        Returns:
            Pair-energy array of shape ``(capacity, capacity)``.

        Raises:
            NotImplementedError: Always, until a subclass supplies the energy.
        """
        raise NotImplementedError(f'{type(self).__name__} must implement pair_energy')

    def pair_params(self, state):
        """Construct per-pair parameter arrays from state.

        Args:
            state: State supplying any field-based couplings.

        Returns:
            Tuple of arrays, each with shape ``(capacity, capacity)``.

        Raises:
            NotImplementedError: Always, until a subclass supplies the parameterization.
        """
        raise NotImplementedError(f'{type(self).__name__} must implement pair_params')

    def mix(self, v):
        """Combine a per-cell coupling ``v`` (N,) into a per-pair coupling (N, N): the arithmetic mean.

        Override for a different symmetric rule (e.g. geometric mean / Lorentz-Berthelot). The
        arithmetic mean has a finite gradient everywhere (no ``sqrt``), so it stays safe at zeros and
        dead cells under ``debug_nans``. The contact distance keeps its own additive rule
        ``sigma = r_i + r_j``.

        Args:
            v: Per-cell scalar coupling array of shape ``(capacity,)``.

        Returns:
            Symmetric pair coupling array of shape ``(capacity, capacity)``.
        """
        return 0.5 * (v[:, None] + v[None, :])

    def _couplings(self):
        """This potential's tunable coupling params, in declaration order (default: none)."""
        return ()

    def _coupling(self, param, state):
        """Resolve a coupling to a per-pair (N, N) matrix: a scalar broadcasts; a spec reads + mixes.

        ``param`` is a shared scalar (Python float / scalar ``jax.Array``) or a ``StateFieldSpec``
        naming a per-cell field that an upstream step or the initial condition sets.
        """
        if isinstance(param, StateFieldSpec):
            return self.mix(getattr(state, param.name))
        n = state.position.shape[0]
        return jnp.broadcast_to(jnp.asarray(param), (n, n))

    def state_reads(self):
        """The per-cell state fields this potential sources its couplings from (the spec params)."""
        return tuple(p for p in self._couplings() if isinstance(p, StateFieldSpec))

    def _check_config(self):
        """Subclass hook for construction-time config validation (default: none)."""

    def __check_init__(self):
        """Validate couplings (a shared scalar or a per-cell scalar spec) and subclass config."""
        for c in self._couplings():
            if isinstance(c, StateFieldSpec):
                if c.shape != ():
                    raise ValueError(
                        f'{type(self).__name__} coupling {c.name!r} must be a per-cell scalar field '
                        f'(a StateFieldSpec of shape ()), got shape {c.shape}.'
                    )
            elif c is not None and jnp.asarray(c).ndim != 0:
                raise ValueError(
                    f'{type(self).__name__} coupling must be a shared scalar or a per-cell '
                    f'StateFieldSpec, not an array of shape {jnp.asarray(c).shape} - per-type / '
                    f'per-cell arrays are not accepted; pass a StateFieldSpec for per-cell values '
                    f'(combined per pair by mix).'
                )
        self._check_config()

    def total_energy(self, positions, state):
        """Sum pair energy over live, non-self pairs, counting each pair once.

        Args:
            positions: Cell positions of shape ``(capacity, n_space_dim)``.
            state: State supplying liveness, space, radii, and any couplings.

        Returns:
            Scalar total pair energy.
        """
        r = safe_norm(pairwise_displacements(positions, state.displacement), axis=-1)
        u = jax.vmap(jax.vmap(self.pair_energy))(r, *self.pair_params(state))
        return 0.5 * jnp.sum(neighbor_sum(u, state.alive))

    def virial_pressure(self, state):
        """Per-cell virial pressure ``p_i = -(1 / (2 d V_i)) sum_j r_ij (dU/dr)(r_ij)``.

        The Irving-Kirkwood one-half bond split (each pair's virial is shared between its two cells);
        the minus sign makes repulsion (``dU/dr < 0``, compression) give ``p > 0``. ``V_i`` is the cell's
        d-ball volume (``2 r``, ``pi r**2``, ``4/3 pi r**3`` for ``d = 1, 2, 3``). Dead cells score 0.

        Args:
            state: State supplying cell geometry, liveness, space, and couplings.

        Returns:
            Per-cell pressure array of shape ``(capacity,)``.
        """
        r = safe_norm(pairwise_displacements(state.position, state.displacement), axis=-1)
        du = jax.grad(self.pair_energy, argnums=0)  # dU/dr, elementwise
        du_dr = jax.vmap(jax.vmap(du))(r, *self.pair_params(state))
        virial = neighbor_sum(r * du_dr, state.alive)  # sum_j r_ij (dU/dr)(r_ij)
        d = state.n_space_dim
        if d == 1:
            volume = 2.0 * state.radius  # a 1-ball is an interval of length 2r
        elif d == 2:
            volume = jnp.pi * state.radius**2  # disk area
        else:  # d == 3
            volume = 4.0 / 3.0 * jnp.pi * state.radius**3  # sphere volume
        return safe_divide(-virial, 2.0 * d * volume) * state.alive.astype(r.dtype)


class Morse(PairwisePotential):
    r"""Morse pair potential with a sigma-relative smooth cutoff.

    With well depth $\epsilon$, steepness $\alpha$, and contact distance $\sigma = r_i + r_j$ (the
    well minimum, where $U = -\epsilon$), the energy is

    $$U(r) = \epsilon\left[\left(1 - e^{-\alpha (r - \sigma)}\right)^2 - 1\right],$$

    multiplied by a smooth cutoff that turns the energy off between
    ``r_onset_frac * sigma`` and ``r_cutoff_frac * sigma``. Larger ``alpha`` narrows the well.
    ``epsilon`` and ``alpha`` are each a shared scalar or a per-cell ``StateFieldSpec``.

    Attributes:
        epsilon: Well depth, scalar or per-cell `StateFieldSpec`. Defaults to 3.0.
        alpha: Well steepness (larger values narrow the well), scalar or per-cell `StateFieldSpec`. Defaults to 2.8.
        r_onset_frac: Smooth-cutoff onset as a multiple of contact distance. Defaults to 1.5.
        r_cutoff_frac: Smooth-cutoff end as a multiple of contact distance. Defaults to 2.5.
    """

    epsilon: object = 3.0  # well depth: float | jax.Array | StateFieldSpec
    alpha: object = 2.8  # well width: float | jax.Array | StateFieldSpec
    r_onset_frac: float = 1.5  # plain numeric fields (cutoff onset / end as multiples of sigma)
    r_cutoff_frac: float = 2.5

    def _couplings(self):
        return (self.epsilon, self.alpha)

    def _check_config(self):
        if not self.r_onset_frac < self.r_cutoff_frac:
            raise ValueError(
                f'Morse needs r_onset_frac < r_cutoff_frac for a smooth cutoff window, '
                f'got r_onset_frac={self.r_onset_frac}, r_cutoff_frac={self.r_cutoff_frac}.'
            )

    def pair_params(self, state):
        """Return ``(sigma, epsilon, alpha)`` as (N, N) arrays; sigma = r_i + r_j is per pair."""
        sigma = state.radius[:, None] + state.radius[None, :]
        return sigma, self._coupling(self.epsilon, state), self._coupling(self.alpha, state)

    def pair_energy(self, r, sigma, eps, alpha):
        """Morse energy at separation ``r`` times the sigma-relative smooth cutoff."""
        e = 1.0 - jnp.exp(-alpha * (r - sigma))
        u = eps * (e * e - 1.0)
        return u * _smooth_cutoff(r, self.r_onset_frac * sigma, self.r_cutoff_frac * sigma)


class SoftSphere(PairwisePotential):
    r"""Harmonic soft-sphere repulsion: a purely repulsive, compact excluded-volume interaction.

    Writing $\sigma = r_i + r_j$ for the contact distance, the pair energy is

    $$U(r) = \frac{\epsilon}{2}\left(1 - \frac{r}{\sigma}\right)^2$$

    for $r < \sigma$ and zero at and beyond contact, so both energy and force vanish there (C1) - the
    canonical soft-disk / active-matter excluded-volume model. ``epsilon`` is a shared scalar or a
    per-cell ``StateFieldSpec``; there is no cutoff to set.

    Attributes:
        epsilon: Repulsion strength, scalar or per-cell `StateFieldSpec`. Defaults to 1.0.
    """

    epsilon: object = 1.0  # repulsion strength: float | jax.Array | StateFieldSpec

    def _couplings(self):
        return (self.epsilon,)

    def pair_params(self, state):
        """Return ``(sigma, epsilon)`` as (N, N) arrays; sigma = r_i + r_j is per pair."""
        sigma = state.radius[:, None] + state.radius[None, :]
        return sigma, self._coupling(self.epsilon, state)

    def pair_energy(self, r, sigma, eps):
        """Harmonic overlap energy, compact at the contact distance."""
        return _compact_repulsion(r, sigma, eps, 2.0, 0.5)


class Hertzian(PairwisePotential):
    r"""Hertzian elastic-contact repulsion: purely repulsive and compact, with a soft onset.

    With stiffness $\epsilon$ and contact distance $\sigma = r_i + r_j$, the pair energy is

    $$U(r) = \frac{2}{5}\,\epsilon \left(1 - \frac{r}{\sigma}\right)^{5/2}$$

    for $r < \sigma$ and zero beyond. Like ``SoftSphere`` but softer at contact - both the force and
    its slope vanish there - modelling deformable elastic spheres / cells. ``epsilon`` is a shared
    scalar or a per-cell ``StateFieldSpec``.

    Attributes:
        epsilon: Contact stiffness, scalar or per-cell `StateFieldSpec`. Defaults to 1.0.
    """

    epsilon: object = 1.0  # contact stiffness: float | jax.Array | StateFieldSpec

    def _couplings(self):
        return (self.epsilon,)

    def pair_params(self, state):
        """Return ``(sigma, epsilon)`` as (N, N) arrays; sigma = r_i + r_j is per pair."""
        sigma = state.radius[:, None] + state.radius[None, :]
        return sigma, self._coupling(self.epsilon, state)

    def pair_energy(self, r, sigma, eps):
        """Hertzian contact energy (exponent 5/2), compact at the contact distance."""
        return _compact_repulsion(r, sigma, eps, 2.5, 0.4)


class Harmonic(PairwisePotential):
    r"""Finite-range harmonic spring: a shifted parabola with its minimum at contact.

    Writing $k$ for the stiffness, $\sigma = r_i + r_j$ for the contact distance, and $r_c$ for the
    cutoff ``r_cutoff_frac * sigma``, the pair energy is

    $$U(r) = \frac{k}{2}\left[(r - \sigma)^2 - (r_c - \sigma)^2\right]$$

    for $r < r_c$ and zero beyond - the parabola shifted down so it vanishes at the cutoff. It is a
    well of depth $\frac{k}{2}(r_c - \sigma)^2$ minimised at contact: repulsive when compressed,
    adhesive when stretched. The energy is only C0 at the cutoff (the force jumps there), harmless
    because the cutoff sits well beyond the resting contact distance. ``k`` is a shared scalar or a
    per-cell ``StateFieldSpec``.

    Attributes:
        k: Spring stiffness, scalar or per-cell `StateFieldSpec`. Defaults to 1.0.
        r_cutoff_frac: Cutoff distance as a multiple of contact distance. Defaults to 2.5.
    """

    k: object = 1.0  # spring stiffness: float | jax.Array | StateFieldSpec
    r_cutoff_frac: float = 2.5  # interaction range as a multiple of sigma (force vanishes beyond)

    def _couplings(self):
        return (self.k,)

    def _check_config(self):
        if not self.r_cutoff_frac > 1.0:
            raise ValueError(
                f'Harmonic needs r_cutoff_frac > 1 (the clamp must sit beyond contact), '
                f'got r_cutoff_frac={self.r_cutoff_frac}.'
            )

    def pair_params(self, state):
        """Return ``(sigma, k)`` as (N, N) arrays; sigma = r_i + r_j is per pair."""
        sigma = state.radius[:, None] + state.radius[None, :]
        return sigma, self._coupling(self.k, state)

    def pair_energy(self, r, sigma, k):
        """Shifted harmonic well: negative inside the cutoff, 0 at and beyond it (truncated)."""
        r_cut = self.r_cutoff_frac * sigma
        u = 0.5 * k * ((r - sigma) ** 2 - (r_cut - sigma) ** 2)  # 0 at r_cut, negative inside
        return jnp.where(r < r_cut, u, 0.0)


class LennardJones(PairwisePotential):
    r"""Lennard-Jones potential (r_min form): minimum ``-epsilon`` at contact, with a sigma-relative cutoff.

    With well depth $\epsilon$ and contact distance $\sigma = r_i + r_j$, the pair energy is

    $$U(r) = \epsilon\left[\left(\frac{\sigma}{r}\right)^{12} - 2\left(\frac{\sigma}{r}\right)^{6}\right],$$

    a hard $r^{-12}$ core plus an $r^{-6}$ adhesive tail with its minimum $-\epsilon$ exactly at
    contact. A smooth cutoff truncates the tail between ``r_onset_frac * sigma`` and
    ``r_cutoff_frac * sigma``. ``epsilon`` is a shared scalar or a per-cell ``StateFieldSpec``.

    Attributes:
        epsilon: Well depth, scalar or per-cell `StateFieldSpec`. Defaults to 1.0.
        r_onset_frac: Smooth-cutoff onset as a multiple of contact distance. Defaults to 1.5.
        r_cutoff_frac: Smooth-cutoff end as a multiple of contact distance. Defaults to 2.5.
    """

    epsilon: object = 1.0  # well depth: float | jax.Array | StateFieldSpec
    r_onset_frac: float = 1.5  # plain numeric fields (cutoff onset / end as multiples of sigma)
    r_cutoff_frac: float = 2.5

    def _couplings(self):
        return (self.epsilon,)

    def _check_config(self):
        if not self.r_onset_frac < self.r_cutoff_frac:
            raise ValueError(
                f'LennardJones needs r_onset_frac < r_cutoff_frac for a smooth cutoff window, '
                f'got r_onset_frac={self.r_onset_frac}, r_cutoff_frac={self.r_cutoff_frac}.'
            )

    def pair_params(self, state):
        """Return ``(sigma, epsilon)`` as (N, N) arrays; sigma = r_i + r_j is per pair."""
        sigma = state.radius[:, None] + state.radius[None, :]
        return sigma, self._coupling(self.epsilon, state)

    def pair_energy(self, r, sigma, eps):
        """LJ energy (r_min form) times the sigma-relative smooth cutoff; safe at r=0."""
        x = safe_divide(sigma, r)  # 0 at the r=0 self-diagonal / dead-dead pairs (no inf)
        u = eps * (x**12 - 2.0 * x**6)
        return u * _smooth_cutoff(r, self.r_onset_frac * sigma, self.r_cutoff_frac * sigma)
