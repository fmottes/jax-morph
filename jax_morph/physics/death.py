"""Cell death as a discrete stochastic removal step.

Forward death draws are exact Bernoulli events. The recorded ``died`` trace uses a backward-only
straight-through surrogate, matching ``Division``. Its physical effect is nevertheless hard:
``alive`` is boolean and the persistent ``death`` record is derived from the thresholded event, so
objectives based on survival or death use the score-function gradient through ``Death.logp``.

When composed with ``Division``, place ``Division`` first and ``Death`` second: the discrete phase
then performs divide-then-die (mothers and newborn daughters may die that macro-step, while slots
freed by death are not recycled until a later one), which keeps the per-step death lineage record
intact.
"""

import jax.numpy as jnp

from ..core.ad_utils import bernoulli_logp, sample_bernoulli_st
from ..core.state import ALIVE, StateFieldSpec
from ..core.step import StepType, StochasticStep

__all__ = ['DEATH_RATE', 'Death']


DEATH_RATE = StateFieldSpec('death_rate', heritable=True)


class Death(StochasticStep):
    r"""Discrete stochastic removal: each alive cell dies as an independent Bernoulli event.

    An alive cell is removed over a macro-step with probability

    $$p = 1 - e^{-\lambda\,\Delta t},$$

    where $\lambda$ is the per-cell ``death_rate``. ``death`` records each slot removed during the
    macro-step for lineage postprocessing, while ``died`` and ``die_eligible`` are the ephemeral
    scored trace. When composed with ``Division``, this step must follow it: the discrete phase
    performs divide-then-die, and deferring reuse of slots freed by death until the next macro-step
    preserves the death lineage record.

    Attributes:
        score_by_default: Whether default trajectory scoring includes death. Defaults to True.

    Methods:
        sample_trace: Draw death actions for eligible cells.
        replay: Apply the recorded removal events.
        logp: Score recorded death actions.
    """

    step_type = StepType.DISCRETE

    def __init__(self, *, score_by_default=True):
        """Build the removal step with its default scoring selection.

        Args:
            score_by_default: Whether default trajectory scoring includes this step. Defaults to
                True.
        """
        self.score_by_default = score_by_default

    def state_reads(self):
        """Read the per-cell death hazard rate."""
        return (DEATH_RATE,)

    def state_writes(self):
        """Write the alive mask and persistent per-step death record."""
        return (ALIVE, StateFieldSpec('death', heritable=False))

    def trace_writes(self):
        """Record the scored action and alive-at-decision eligibility mask."""
        return (
            StateFieldSpec('died', heritable=False),
            StateFieldSpec('die_eligible', heritable=False),
        )

    def _dist(self, state, dt):
        """Return the exact per-cell hazard probability used for sampling and scoring."""
        rate = jnp.clip(state.death_rate, 0.0, None)
        return -jnp.expm1(-rate * dt)

    def sample_trace(self, state, *, dt, key):
        """Draw forward-exact, backward-straight-through death actions.

        Args:
            state: Pre-step state with ``alive`` and ``death_rate`` arrays of shape ``(capacity,)``.
            dt: Macro-step duration.
            key: JAX PRNG key.

        Returns:
            Trace dictionary with ``died`` and ``die_eligible`` arrays of shape ``(capacity,)``.
        """
        alive_f = state.alive.astype(state.death_rate.dtype)
        died = sample_bernoulli_st(key, self._dist(state, dt)) * alive_f
        return {'died': died, 'die_eligible': alive_f}

    def replay(self, state, trace, *, dt, pathwise):
        """Apply recorded hard death actions and co-emit their trace.

        Args:
            state: Pre-step state.
            trace: Recorded death action and eligibility arrays.
            dt: Unused macro-step duration.
            pathwise: Unused because the discrete event replays its recorded action.

        Returns:
            Post-death state with ``alive``, ``death``, and trace fields updated.
        """
        die = (trace['died'] > 0.5) & state.alive
        alive = state.alive & ~die
        death = die.astype(state.death_rate.dtype)
        return state.update(
            alive=alive,
            death=death,
            died=trace['died'],
            die_eligible=trace['die_eligible'],
        )

    def logp(self, state, trace, dt):
        """Score recorded actions over cells alive at the death decision.

        Args:
            state: Live pre-step state conditioning death probabilities.
            trace: Recorded action and eligibility arrays.
            dt: Macro-step duration.

        Returns:
            Scalar sum of eligible-cell Bernoulli log-probabilities.
        """
        return jnp.sum(bernoulli_logp(trace['died'], self._dist(state, dt)) * trace['die_eligible'])
