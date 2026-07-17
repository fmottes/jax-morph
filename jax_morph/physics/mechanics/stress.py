"""Per-cell virial stress: a quasistatic sensing input derived from a pair potential.

``VirialStress`` writes each cell's virial pressure into a ``stress`` field, so a downstream control
step can sense the local mechanical load. The pressure itself is computed by the potential
(``PairwisePotential.virial_pressure``); this step only exposes it as a state field each macro-step.
"""

from ...core.state import ALIVE, POSITION, RADIUS, StateFieldSpec
from ...core.step import SimulationStep, StepType
from .potentials import PairwisePotential

__all__ = ['VirialStress']


class VirialStress(SimulationStep):
    r"""Quasistatic step writing each cell's virial pressure into a per-cell ``stress`` field.

    For any ``PairwisePotential`` the per-cell value is the Irving-Kirkwood virial pressure

    $$p_i = -\frac{1}{2 d V_i} \sum_j r_{ij}\, \frac{dU}{dr}(r_{ij}),$$

    where $d$ is the spatial dimension, $V_i$ the cell's $d$-ball volume, and the one-half shares each
    pair's virial between its two cells. The minus sign makes compression (repulsion,
    $\frac{dU}{dr} < 0$) read positive and tension negative; cells beyond the potential's cutoff read
    0. ``stress`` is a transient sensing quantity (``heritable=False``), recomputed from the live
    configuration each macro-step; the step's parameters ride inside ``potential``, so a traced
    potential parameter (e.g. a ``jax.Array`` ``epsilon``) is optimizable through the written stress.

    Attributes:
        potential: Pairwise interaction potential used to compute virial pressure.

    Methods:
        __call__: Compute and store per-cell stress.
    """

    step_type = StepType.QUASISTATIC
    potential: PairwisePotential  # the pair potential whose virial_pressure is exposed

    def state_reads(self):
        """Reads positions, radii, the alive mask, and any field the potential sources params from."""
        return (POSITION, RADIUS, ALIVE, *self.potential.state_reads())

    def state_writes(self):
        """Writes the per-cell ``stress`` field (a transient sensing quantity)."""
        return (StateFieldSpec('stress', heritable=False),)

    def __call__(self, state, *, dt, key):
        """Set ``stress`` to the potential's per-cell virial pressure.

        Args:
            state: Input state with cell fields required by the potential.
            dt: Unused macro-step duration.
            key: Unused JAX PRNG key.

        Returns:
            State with ``stress`` set to an array of shape ``(capacity,)``.
        """
        return state.set('stress', self.potential.virial_pressure(state))
