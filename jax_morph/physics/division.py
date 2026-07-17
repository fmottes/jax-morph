r"""Cell division as a discrete jump: volume-conserving, oriented, non-overlapping daughters.

The forward draw is an exact Bernoulli event with a backward-only straight-through surrogate, so the
sampled trajectory is a true sample. Two gradient estimators come from one forward, as alternatives
(pick one per run, never sum them): the score-function gradient via ``Division.logp`` (unbiased for
the discrete event) and a pathwise gradient from differentiating ``simulate`` (the surrogate reaches
the hazard rate, the reparameterized placement reaches the orientation parameters, biased through the
discrete slot allocation).

Placement is oriented along $s\,\hat{a} + \xi/\sqrt{d}$ (then normalized), where $\hat{a}$ is an
optional per-cell ``division_axis``, $\xi$ is isotropic noise, and $s$ = ``orientation_snr`` is a
dimension-independent amplitude signal-to-noise ratio ($s = 0$ or a zero axis gives isotropic
placement). Lineage is recorded minimally as ``born`` (new-daughter flag) and ``mother`` (parent slot
index); ``reconstruct_lineage`` rebuilds a stable lineage graph in postprocessing. Capacity overflow
is capped into the global ``division_overflow`` counter, never crashing a jit / vmap / scan batch.

When composed with ``Death``, place ``Division`` first and ``Death`` second: the discrete phase then
performs divide-then-die (mothers and newborn daughters may die that macro-step, while slots freed by
death are not recycled until a later one), which keeps the per-step death lineage record intact.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ..core.ad_utils import bernoulli_logp, safe_norm, sample_bernoulli_st
from ..core.state import ALIVE, CELLTYPE, POSITION, RADIUS, StateFieldSpec
from ..core.step import StepType, StochasticStep

__all__ = ['DIVISION_RATE', 'Division', 'reconstruct_lineage']

# Per-cell hazard rate: heritable (a daughter inherits its propensity; a controller that rewrites it
# each macro-step makes that moot). Exported so a control step can declare the SAME spec.
DIVISION_RATE = StateFieldSpec('division_rate', heritable=True)


class Division(StochasticStep):
    r"""Discrete stochastic step: each alive cell divides as an independent Bernoulli event.

    A cell divides over a macro-step with probability

    $$p = 1 - e^{-\lambda\,\Delta t},$$

    where $\lambda$ is the per-cell ``division_rate`` (a constant initial condition, or an upstream
    controller's output - size / crowding / morphogen gating all live there). Daughters conserve
    volume, each taking radius $r\,2^{-1/d}$, and are placed just touching along an orientable
    direction, inheriting the mother's ``heritable`` fields. ``orientation_snr`` (the amplitude
    signal-to-noise ratio; 0 -> isotropic) sharpens placement toward the per-cell ``division_axis``.
    Lineage is recorded as ``born`` + ``mother``; overflow past ``capacity`` is capped into the global
    ``division_overflow`` counter. When composed with ``Death``, this step must precede it: the
    discrete phase performs divide-then-die, and deferring reuse of slots freed by death until the
    next macro-step preserves the death lineage record.

    Attributes:
        n_space_dim: Static spatial dimension used for ``division_axis`` and ``division_dir``, both
            shaped ``(capacity, n_space_dim)``.
        orientation_snr: Placement-orientation signal-to-noise amplitude. Defaults to 0.0.
        score_by_default: Whether default trajectory scoring includes division. Defaults to True.

    Methods:
        sample_trace: Draw division actions and placement directions.
        replay: Apply the recorded division events.
        logp: Score recorded division actions.
    """

    step_type = StepType.DISCRETE
    n_space_dim: int = eqx.field(static=True)  # sets the (n_space_dim,) shape of the vector fields
    orientation_snr: object  # amplitude SNR; 0 -> isotropic. Python float static / Array traced

    def __init__(self, *, n_space_dim, orientation_snr=0.0, score_by_default=True):
        """Build the step from the spatial dimension and orientation signal-to-noise ratio.

        Args:
            n_space_dim: Spatial dimension; must be 1, 2, or 3.
            orientation_snr: Placement-orientation signal-to-noise amplitude. Defaults to 0.0.
            score_by_default: Whether default trajectory scoring includes this step. Defaults to
                True.

        Raises:
            ValueError: If ``n_space_dim`` is not 1, 2, or 3.
        """
        if n_space_dim not in (1, 2, 3):
            raise ValueError(f'Division n_space_dim must be 1, 2, or 3, got {n_space_dim}.')
        self.n_space_dim = n_space_dim
        self.orientation_snr = orientation_snr
        self.score_by_default = score_by_default

    def state_reads(self):
        """Reads the per-cell hazard rate and the optional per-cell orientation axis."""
        return (
            DIVISION_RATE,
            StateFieldSpec('division_axis', shape=(self.n_space_dim,), heritable=True),
        )

    def state_writes(self):
        """Writes the base fields, the overflow diagnostic, and the lineage fields."""
        return (
            POSITION,
            RADIUS,
            CELLTYPE,
            ALIVE,
            StateFieldSpec('division_overflow', scope='global', heritable=False),
            StateFieldSpec('born', heritable=False),
            StateFieldSpec('mother', dtype=int, default=-1, heritable=False),
        )

    def trace_writes(self):
        """Ephemeral trace: the 0/1 action, the eligibility mask, and the realized direction."""
        return (
            StateFieldSpec('divided', heritable=False),
            StateFieldSpec('divide_eligible', heritable=False),
            StateFieldSpec('division_dir', shape=(self.n_space_dim,), heritable=False),
        )

    def _dist(self, state, dt):
        """Per-cell probability ``p = 1 - exp(-lambda*dt)`` (single source for sample and score)."""
        rate = jnp.clip(state.division_rate, 0.0, None)
        return -jnp.expm1(-rate * dt)

    def sample_trace(self, state, *, dt, key):
        """Draw division actions and oriented placement directions.

        Args:
            state: Pre-step state with cell arrays of leading shape ``(capacity,)``.
            dt: Macro-step duration.
            key: JAX PRNG key.

        Returns:
            Trace dictionary containing ``divided`` and ``divide_eligible`` arrays of shape
            ``(capacity,)`` and ``division_dir`` of shape ``(capacity, n_space_dim)``.
        """
        key_div, key_dir = jax.random.split(key)
        alive_f = state.alive.astype(state.radius.dtype)
        divided = sample_bernoulli_st(key_div, self._dist(state, dt)) * alive_f
        axis_hat = state.division_axis / (
            safe_norm(state.division_axis, axis=-1, keepdims=True) + 1e-12
        )
        # noise / sqrt(d) has unit RMS magnitude, so orientation_snr is a dimension-independent
        # amplitude signal-to-noise ratio (aligned amplitude vs unit-RMS isotropic noise).
        xi = jax.random.normal(key_dir, (state.n_cells, self.n_space_dim)) / jnp.sqrt(
            self.n_space_dim
        )
        biased = self.orientation_snr * axis_hat + xi
        direction = biased / (safe_norm(biased, axis=-1, keepdims=True) + 1e-12)
        return {'divided': divided, 'divide_eligible': alive_f, 'division_dir': direction}

    def replay(self, state, trace, *, dt, pathwise):
        """Apply recorded division events and co-emit their trace.

        Args:
            state: Pre-step state.
            trace: Recorded action, eligibility, and direction arrays.
            dt: Unused macro-step duration.
            pathwise: Unused because the discrete event always replays its recorded action.

        Returns:
            Post-division state with physical outputs and trace fields updated.

        Raises:
            ValueError: If the state's spatial dimension differs from ``n_space_dim``.
        """
        if state.position.shape[1] != self.n_space_dim:
            raise ValueError(
                f'Division was built with n_space_dim={self.n_space_dim} but the state is '
                f'{state.position.shape[1]}-D; they must match.'
            )
        n, d = state.n_cells, self.n_space_dim
        m = 2.0 ** (-1.0 / d)
        divided_soft = trace['divided']  # soft 0/1: carries the forward straight-through gradient
        divide = divided_soft > 0.5  # hard decision drives the discrete slot allocation

        # assign the k-th committed divider to the k-th free slot (n == "no slot" -> dropped)
        free_idx = jnp.nonzero(~state.alive, size=n, fill_value=n)[0]
        div_rank = jnp.cumsum(divide) - 1
        new_slot = jnp.where(divide, free_idx[jnp.clip(div_rank, 0, n - 1)], n)
        committed = divide & (new_slot < n)
        committed_soft = divided_soft * committed.astype(divided_soft.dtype)

        # daughter offset uses the NEW radius r' = r*m, along the recorded direction
        offset = (state.radius * m)[:, None] * trace['division_dir']
        radius = state.radius * (1.0 - committed_soft * (1.0 - m))
        radius = radius.at[new_slot].set(state.radius * m, mode='drop')
        position = state.position + committed_soft[:, None] * offset
        position = position.at[new_slot].set(state.position - offset, mode='drop')
        alive = state.alive.at[new_slot].set(True, mode='drop')
        state = state.update(radius=radius, position=position, alive=alive)

        # Give each daughter its cell-scope fields: heritable ones inherit the mother's value, the
        # rest are (re)initialised to their default - the free slot may be a recycled dead cell still
        # holding stale values, and the StateFieldSpec contract is that a non-heritable field starts
        # at its default on a newborn daughter. Base fields (position/radius/alive) are set above and
        # globals (grids) are not per-cell, so both are skipped.
        for name, spec in state.specs.items():
            if name in ('position', 'radius', 'alive') or spec.scope != 'cell':
                continue
            val = state[name]
            fill = val if spec.heritable else spec.default
            state = state.set(name, val.at[new_slot].set(fill, mode='drop'))

        # lineage: born flag on daughter slots, mother = parent slot index (sentinel elsewhere)
        born = jnp.zeros(n, state.radius.dtype).at[new_slot].set(1.0, mode='drop')
        mother = jnp.full((n,), -1, state.mother.dtype).at[new_slot].set(jnp.arange(n), mode='drop')
        n_dropped = jnp.sum(divide) - jnp.sum(committed)
        # co-emit the trace fields (so scoring reads them back) and accumulate the capped overflow
        return state.update(
            born=born,
            mother=mother,
            divided=trace['divided'],
            divide_eligible=trace['divide_eligible'],
            division_dir=trace['division_dir'],
            division_overflow=state.division_overflow
            + n_dropped.astype(state.division_overflow.dtype),
        )

    def logp(self, state, trace, dt):
        """Score recorded division actions over cells alive at the decision.

        Args:
            state: Live pre-step state conditioning division probabilities.
            trace: Recorded action and eligibility arrays.
            dt: Macro-step duration.

        Returns:
            Scalar sum of eligible-cell Bernoulli log-probabilities.
        """
        return jnp.sum(
            bernoulli_logp(trace['divided'], self._dist(state, dt)) * trace['divide_eligible']
        )


def reconstruct_lineage(born, mother, alive, death=None):
    """Rebuild a cell-lineage graph from a recorded simulation history (postprocessing, not jitted).

    A cell's identity in a running simulation is its state-array slot, and slots are reused after a
    cell dies, so a stable lineage needs the per-step birth record. A complete simulation history
    begins with its initial state: every slot alive in frame zero is a founder, and every later
    ``born`` slot mints a fresh node whose parent is the node currently occupying the recorded
    ``mother`` slot. When a matching ``death`` history is provided, each node is annotated with the
    macro-step when it was removed. Births are processed before deaths within a frame, matching the
    required ``Division``-then-``Death`` order.

    Args:
        born: ``(n_steps + 1, capacity)`` new-daughter flags over a complete history.
        mother: ``(n_steps + 1, capacity)`` parent slot indices (sentinel -1).
        alive: ``(n_steps + 1, capacity)`` alive masks (frame zero seeds founders).
        death: Optional ``(n_steps + 1, capacity)`` per-step death flags. Defaults to None.

    Returns:
        A list of nodes ordered by id, each a dict with ``id`` (unique int), ``parent`` (parent id, or
        ``None`` for a founder), ``slot`` (the array slot it was born into), ``birth_step``, and
        ``death_step`` (or ``None`` when no death was recorded).
    """
    born = np.asarray(born)
    mother = np.asarray(mother)
    alive = np.asarray(alive)
    death = None if death is None else np.asarray(death)
    nodes = []
    slot_to_id = {}
    next_id = 0
    for slot in np.flatnonzero(alive[0]):
        nodes.append(
            {'id': next_id, 'parent': None, 'slot': int(slot), 'birth_step': 0, 'death_step': None}
        )
        slot_to_id[int(slot)] = next_id
        next_id += 1
    for t in range(1, born.shape[0]):
        for slot in np.flatnonzero(born[t] > 0.5):
            parent = slot_to_id.get(int(mother[t, slot]))
            nodes.append(
                {
                    'id': next_id,
                    'parent': parent,
                    'slot': int(slot),
                    'birth_step': int(t),
                    'death_step': None,
                }
            )
            slot_to_id[int(slot)] = next_id
            next_id += 1
        if death is not None:
            for slot in np.flatnonzero(death[t] > 0.5):
                node_id = slot_to_id.pop(int(slot), None)
                if node_id is not None:
                    nodes[node_id]['death_step'] = int(t)
    return nodes
