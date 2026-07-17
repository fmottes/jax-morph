"""Steady-state screened diffusion via dimension-correct, finite-size analytic Green's functions.

Free-space (open-boundary) model: the kernel is the unbounded screened-diffusion Green's function, so
open boundaries are automatic - pairing this step with a bounded / periodic space is a modeling error
the user owns (the periodic screened field is a different, Ewald-like solution, not enforced).
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.special as jsp

from ..core.ad_utils import safe_norm
from ..core.geometry import pairwise_displacements
from ..core.state import StateFieldSpec
from ..core.step import SimulationStep, StepType

__all__ = ['FreeScreenedDiffusion']


def _k0(x):
    """Modified Bessel K0 (Abramowitz and Stegun 9.8.5 / 9.8.6); differentiable, finite for x > 0."""
    xs = jnp.minimum(x, 2.0)
    y = (xs / 2.0) ** 2
    k_small = -jnp.log(xs / 2.0) * jsp.i0(xs) + (
        -0.57721566
        + y
        * (
            0.42278420
            + y
            * (0.23069756 + y * (0.03488590 + y * (0.00262698 + y * (0.00010750 + y * 0.00000740))))
        )
    )
    xl = jnp.maximum(x, 2.0)
    z = 2.0 / xl
    k_large = (
        jnp.exp(-xl)
        / jnp.sqrt(xl)
        * (
            1.25331414
            + z
            * (
                -0.07832358
                + z
                * (
                    0.02189568
                    + z * (-0.01062446 + z * (0.00587872 + z * (-0.00251540 + z * 0.00053208)))
                )
            )
        )
    )
    return jnp.where(x <= 2.0, k_small, k_large)


def _k1(x):
    """Modified Bessel K1 (Abramowitz and Stegun 9.8.7 / 9.8.8); differentiable, finite for x > 0."""
    xs = jnp.minimum(x, 2.0)
    y = (xs / 2.0) ** 2
    k1_small = jnp.log(xs / 2.0) * jsp.i1(xs) + (1.0 / xs) * (
        1.0
        + y
        * (
            0.15443144
            + y
            * (
                -0.67278579
                + y * (-0.18156897 + y * (-0.01919402 + y * (-0.00110404 + y * -0.00004686)))
            )
        )
    )
    xl = jnp.maximum(x, 2.0)
    z = 2.0 / xl
    k1_large = (
        jnp.exp(-xl)
        / jnp.sqrt(xl)
        * (
            1.25331414
            + z
            * (
                0.23498619
                + z
                * (
                    -0.03655620
                    + z * (0.01504268 + z * (-0.00780353 + z * (0.00325614 + z * -0.00068245)))
                )
            )
        )
    )
    return jnp.where(x <= 2.0, k1_small, k1_large)


class FreeScreenedDiffusion(SimulationStep):
    r"""Quasistatic free-space field: superpose each cell's finite-size screened-diffusion Green's function.

    Every alive cell of radius $a = r_i$ - a sphere (3-D), disk (2-D), or segment (1-D) - emits at a
    total rate $S$. The steady concentration solves the screened-diffusion (modified Helmholtz)
    equation

    $$D\,\nabla^2 c - k\,c = -S,$$

    so the field at cell $i$ superposes the per-source open-boundary Green's functions,

    $$c_i = \sum_j S_j\, G(r_{ij}),$$

    with inverse screening length $\kappa = \sqrt{k / D}$ ($D$ = ``diffusion``, $k$ = ``degradation``).
    The near / self field is regularized by clamping the receiver distance to the source surface,
    $r \to \max(r, a)$.

    Field-agnostic: ``source_name`` / ``field_name`` name the state fields read (per-cell source rate)
    and written (the resulting field), defaulting to ``secretion_rate`` / ``chemical``. Override them
    to model any diffusing quantity (heat, a signaling molecule, ...).

    ``diffusion`` and ``degradation`` are shared scalars or per-species arrays of shape
    ``(n_field_species,)``. Screening (``degradation > 0``) is required in 1-D and 2-D, where the
    unscreened field is unbounded; 3-D admits ``degradation = 0``.

    Attributes:
        n_field_species: Static number of chemical species; source and field arrays have shape
            ``(capacity, n_field_species)``.
        n_space_dim: Static spatial dimension selecting the segment, disk, or sphere kernel.
        diffusion: Diffusion coefficient, scalar or shape ``(n_field_species,)``. Defaults to 1.0.
        degradation: Degradation coefficient, scalar or shape ``(n_field_species,)``. Defaults to
            1.0.
        source_name: Name of the per-cell source-rate state field. Defaults to ``'secretion_rate'``.
        field_name: Name of the computed per-cell field. Defaults to ``'chemical'``.
    """

    step_type = StepType.QUASISTATIC
    n_field_species: int = eqx.field(
        static=True
    )  # number of field species -> shape-determining, static
    n_space_dim: int = eqx.field(
        static=True
    )  # 1 (segment), 2 (disk), 3 (sphere) -> selects the kernel
    source_name: str = eqx.field(static=True)  # per-cell source-rate field this step reads
    field_name: str = eqx.field(static=True)  # per-cell field this step writes
    diffusion: object  # plain numeric fields: Python float -> static, jax.Array -> traced
    degradation: object

    def __init__(
        self,
        *,
        n_field_species,
        n_space_dim,
        diffusion=1.0,
        degradation=1.0,
        source_name='secretion_rate',
        field_name='chemical',
    ):
        """Build the step from field dimensions, names, and physical coefficients.

        Args:
            n_field_species: Number of field species.
            n_space_dim: Spatial dimension; must be 1, 2, or 3.
            diffusion: Diffusion coefficient, scalar or per-species array. Defaults to 1.0.
            degradation: Degradation coefficient, scalar or per-species array. Defaults to 1.0.
            source_name: Per-cell source-rate field name. Defaults to ``'secretion_rate'``.
            field_name: Per-cell output field name. Defaults to ``'chemical'``.

        Raises:
            ValueError: If the spatial dimension is unsupported or a static low-dimensional
                degradation is nonpositive.
        """
        if n_space_dim not in (1, 2, 3):
            raise ValueError(
                f'FreeScreenedDiffusion n_space_dim must be 1 (segment), 2 (disk), or 3 (sphere), '
                f'got {n_space_dim}.'
            )
        # In 1-D and 2-D the unscreened steady field is unbounded (kappa -> 0 diverges), so a
        # statically zero / negative degradation is a modeling error; a traced value is left to run.
        if n_space_dim in (1, 2) and isinstance(degradation, int | float) and degradation <= 0:
            raise ValueError(
                f'FreeScreenedDiffusion in {n_space_dim}-D needs degradation > 0 (kappa = '
                f'sqrt(degradation / diffusion); the unscreened low-dimension field is unbounded).'
            )
        self.n_field_species = n_field_species
        self.n_space_dim = n_space_dim
        self.source_name = source_name
        self.field_name = field_name
        self.diffusion = diffusion
        self.degradation = degradation

    def state_reads(self):
        """Reads each cell's per-species source rate (the ``source_name`` field)."""
        return (StateFieldSpec(self.source_name, shape=(self.n_field_species,)),)

    def state_writes(self):
        """Writes the per-cell field (``field_name``, recomputed each macro-step, not heritable)."""
        return (StateFieldSpec(self.field_name, shape=(self.n_field_species,), heritable=False),)

    def _kernel(self, r, a, kappa, D):
        """Screened Green's function of a finite source of radius ``a``, clamped to its surface.

        Clamp the receiver distance to the source surface (``r_eff = max(r, a)``) so the near / self
        field takes the physical on-surface value. Dead cells carry radius ``a = 0``, which makes the
        self / near field singular (``1 / r_eff`` in 3-D, ``K1(0)`` in 2-D); floor ``a`` so the kernel
        stays finite there. Those entries are zeroed by the ``alive`` source mask in ``__call__``, and
        the floor keeps the singularity from becoming ``inf * 0 -> NaN`` under ``jax_debug_nans``. The
        1-D segment kernel has no geometric singularity (it is finite even at ``a = 0``).

        The 1-D and 2-D kernels divide by ``kappa`` (and evaluate ``K1(kappa a)``), so a screening
        length ``kappa = 0`` - reachable only through a *traced* ``degradation`` that transiently hits
        0 during optimization, since a static zero is rejected at construction - would give ``inf`` /
        ``K1(0) -> NaN``. Floor ``kappa`` for those two kernels: it is a no-op for any real screened
        config (``kappa`` is many orders above the floor) and merely returns a large finite value at the
        degenerate ``kappa = 0`` instead of crashing. The 3-D kernel is bounded at ``kappa = 0`` (it
        reduces to the unscreened ``1 / (4 pi D r_eff)``), so it is left untouched and exact there.

        Args:
            r: Receiver-source distances of shape ``(capacity, capacity)``.
            a: Source radii, broadcastable to ``r``.
            kappa: Inverse screening length for one species.
            D: Diffusion coefficient for one species.

        Returns:
            Green's-function values of shape ``(capacity, capacity)``.
        """
        a = jnp.maximum(a, 1e-12)
        r_eff = jnp.maximum(r, a)
        if self.n_space_dim == 1:  # 1-D finite segment: exp(-kappa (r_eff - a)) / (2 D kappa)
            kappa = jnp.maximum(kappa, 1e-12)
            return jnp.exp(-kappa * (r_eff - a)) / (2.0 * D * kappa)
        if self.n_space_dim == 3:  # 3-D finite sphere (bounded at kappa = 0; no floor needed)
            return jnp.exp(-kappa * (r_eff - a)) / (4.0 * jnp.pi * D * r_eff * (1.0 + kappa * a))
        # 2-D finite disk
        kappa = jnp.maximum(kappa, 1e-12)
        return _k0(kappa * r_eff) / (2.0 * jnp.pi * D * a * kappa * _k1(kappa * a))

    def __call__(self, state, *, dt, key):
        """Set the output field to the superposed steady-state screened-diffusion field.

        Args:
            state: Input state with positions of shape ``(capacity, n_space_dim)`` and source
                field of shape ``(capacity, n_field_species)``.
            dt: Unused macro-step duration.
            key: Unused JAX PRNG key.

        Returns:
            State with ``field_name`` set to an array of shape ``(capacity, n_field_species)``.

        Raises:
            ValueError: If the state's spatial dimension differs from ``n_space_dim``.
        """
        if state.position.shape[1] != self.n_space_dim:
            raise ValueError(
                f'FreeScreenedDiffusion was built with n_space_dim={self.n_space_dim} (it selects the '
                f'segment/disk/sphere kernel) but the state has spatial dimension '
                f'{state.position.shape[1]}; they must match.'
            )
        disp = pairwise_displacements(state.position, state.displacement)
        r = safe_norm(disp, axis=-1)  # (N, N) center-to-center distances
        a = state.radius[None, :]  # (1, N) SOURCE radii (broadcast over the receiver rows i)
        alive_f = state.alive.astype(r.dtype)
        D = jnp.broadcast_to(jnp.asarray(self.diffusion, float), (self.n_field_species,))
        K = jnp.broadcast_to(jnp.asarray(self.degradation, float), (self.n_field_species,))
        S = state[self.source_name]

        def one(Dc, Kc, Sc):
            kappa = jnp.sqrt(Kc / Dc)
            G = self._kernel(r, a, kappa, Dc) * alive_f[None, :]  # only live sources
            return G @ Sc

        field = jax.vmap(one, in_axes=(0, 0, 1), out_axes=1)(D, K, S)
        return state.set(self.field_name, field * alive_f[:, None])
