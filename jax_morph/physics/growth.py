"""Cell growth as finite-rate radius dynamics: a saturating (von Bertalanffy) law as a sparse delta.

The growth rate is a per-cell **state field** (``growth_rate``), not a module parameter, so it can be
a fixed initial-condition constant or - the intended use - written each macro-step by an upstream
control step (a gene network / MLP), with gradients flowing from a morphology objective through the
per-cell rate to whatever produced it. Only the asymptotic target size ``max_radius`` is a module
parameter. Other growth laws can live alongside this one in this module.
"""

import jax.numpy as jnp

from ..core.state import RADIUS, StateFieldSpec
from ..core.step import SimulationStep, StepType

__all__ = ['GROWTH_RATE', 'SaturatingCellGrowth']

# The per-cell growth rate. Heritable: an intrinsic growth propensity a daughter inherits (a
# controller that rewrites it each macro-step makes heritability moot); default 0 means no growth
# unless a rate is supplied. Exported so a control step can declare the SAME spec (merge-compatible).
GROWTH_RATE = StateFieldSpec('growth_rate', heritable=True)


class SaturatingCellGrowth(SimulationStep):
    r"""Dynamic step: grow each cell radius toward ``max_radius`` by a saturating (von Bertalanffy) law.

    Writing $k$ for the per-cell ``growth_rate`` and $R$ for ``max_radius``, the radius obeys

    $$\frac{dr}{dt} = k \left(1 - \frac{r}{R}\right),$$

    so growth is proportional to the remaining gap to the target size (not logistic): fastest at small
    ``r``, relaxing to zero at ``max_radius``, and positive from any birth size. ``growth_rate`` is
    read per cell from the state - a constant initial condition or an upstream controller's output -
    while ``max_radius`` is the module's asymptotic target size.

    The increment applied over a step ``dt`` is the **exact** flow of this linear law, not forward
    Euler:

    $$\Delta r = (R - r) \left(1 - e^{-k\,\Delta t / R}\right),$$

    which is unconditionally stable and monotone for any ``dt`` and any ``growth_rate >= 0``.

    Attributes:
        max_radius: Asymptotic cell radius. A scalar ``jax.Array`` is optimizable. Defaults to 1.0.
    """

    step_type = StepType.DYNAMIC
    max_radius: object  # plain field: Python float -> static, jax.Array -> traced (target size)

    def __init__(self, *, max_radius=1.0):
        """Build the step from the asymptotic target radius.

        Args:
            max_radius: Asymptotic cell radius. Defaults to 1.0.
        """
        self.max_radius = max_radius

    def state_reads(self):
        """Reads the current radius and the per-cell growth rate."""
        return (RADIUS, GROWTH_RATE)

    def state_writes(self):
        """Writes the radius increment (a dynamic delta)."""
        return (RADIUS,)

    def __call__(self, state, *, dt, key):
        """Return the exact saturating-growth radius increment over ``dt`` as a sparse delta.

        ``key`` is unused. The increment is exact while this is the only dynamic writer of ``radius``
        (the case today); with a co-writer it degrades to a stable operator-split contribution, still
        additive.

        Args:
            state: Input state containing ``radius`` and ``growth_rate`` arrays of shape
                ``(capacity,)``.
            dt: Macro-step duration.
            key: Unused JAX PRNG key.

        Returns:
            Sparse dynamic delta with a ``radius`` array of shape ``(capacity,)``.
        """
        decay = jnp.exp(-state.growth_rate * dt / self.max_radius)
        dr = (self.max_radius - state.radius) * (1.0 - decay)
        return state.deltas(radius=dr)
