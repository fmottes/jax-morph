"""Step contract, the validated model composition, and the hybrid macro-step.

A step is an ``eqx.Module`` with one uniform method, ``__call__(state, *, dt, key) -> state``. What
the returned numbers *mean* is interpreted by the model via ``step_type``: a quasistatic or discrete
step returns the new state directly, while a dynamic step returns a sparse *delta* state (its
``dt``-scaled increment, all untouched fields ``None``) that the model accumulates.

Scoring the stochastic choices is a separate concern. A stochastic step is a ``StochasticStep``: it
records a **trace** - the exogenous noise plus the realized action - into declared, ephemeral trace
fields, and owns a single ``replay`` that is the one source of its effect. Forward, ``__call__``
samples a trace and replays it; the top-level scoring drivers read recorded traces and replay them
with parameters live. ``trajectory_logp`` carries the reconstructed state across macro-steps, while
``transition_logp`` detaches an explicitly observed conditioning state. Both use the model's private
single-transition replay machinery and each stochastic step's ``logp`` hook.
"""

from collections.abc import Iterable
from enum import StrEnum

import equinox as eqx
import jax
import jax.numpy as jnp

from .state import BASE_SPECS, BaseState, StateFieldSpec, merge_specs

__all__ = [
    'StepType',
    'SimulationStep',
    'StochasticStep',
    'Model',
    'check_stochastic_step',
]

_PLACEHOLDER_KEY = jax.random.PRNGKey(0)  # deterministic steps ignore the key


def _is_bool_like(x):
    """Whether ``x`` is a boolean scalar - a Python ``bool`` or a numpy/jax bool-dtype scalar.

    Used to reject a boolean mask passed as a ``score=`` selection: ``bool`` subclasses ``int``, so
    a mask would otherwise be silently read as the integer index set ``{0, 1}``. A plain
    ``isinstance(x, bool)`` misses numpy/jax bools.
    """
    if isinstance(x, bool):
        return True
    dtype = getattr(x, 'dtype', None)
    return dtype is not None and jnp.issubdtype(dtype, jnp.bool_)


class StepType(StrEnum):
    """The type of dynamics a step contributes."""

    DYNAMIC = 'dynamic'  # contributes a dt-scaled increment, accumulated across writers
    QUASISTATIC = 'quasistatic'  # instantaneous, recomputed each macro-step
    DISCRETE = 'discrete'  # discrete event


class SimulationStep(eqx.Module):
    """Base class for all simulation steps. Subclasses set the class var ``step_type``.

    A step documents its inputs and outputs as tuples of ``StateFieldSpec`` (each spec carries its
    own name):

    - ``state_reads()`` -> the fields it consumes.
    - ``state_writes()`` -> the fields it produces.

    Use the prefilled base specs (``POSITION``, ``RADIUS``, ...) for base fields; the step does not
    care whether a field is base or who else touches it. ``state_requires()`` is the derived union
    ``state_reads U state_writes``. Consistency and merging across steps is the ``Model``'s job.

    Every step implements the same method, ``__call__(state, *, dt, key) -> state`` (``dt`` and
    ``key`` are keyword-only); its return is interpreted by ``step_type``:

    - **QUASISTATIC / DISCRETE:** return the new state (the constraint solve, or the jump, applied).
    - **DYNAMIC:** return ``state.deltas(**{field: increment})`` - a sparse delta state whose written
      fields hold that field's increment *over ``dt``* (the step bakes in its own ``dt``-scaling) and
      whose every other field is ``None``. The model sums these across dynamic writers, so a dynamic
      step returns only its own contribution, never a full state.

    A deterministic step is exactly this. A stochastic step additionally samples and scores an
    action; it subclasses ``StochasticStep`` (below), which supplies ``is_stochastic = True`` and the
    trace / replay / logp contract.

    Methods:
        state_reads: Declare state fields consumed by this step.
        state_writes: Declare physical state fields produced by this step.
        state_requires: Return the merged read/write schema.
        __call__: Apply the step according to its declared step type.
    """

    step_type: eqx.AbstractClassVar[StepType]

    def state_reads(self) -> tuple:
        """Return field specifications this step reads.

        Returns:
            A tuple of `StateFieldSpec` objects. Defaults to an empty tuple.
        """
        return ()

    def state_writes(self) -> tuple:
        """Return physical field specifications this step writes.

        Returns:
            A tuple of `StateFieldSpec` objects. Defaults to an empty tuple.
        """
        return ()

    def state_requires(self) -> tuple:
        """Return the deduplicated union of read and written field specifications.

        Returns:
            A tuple of `StateFieldSpec` objects, preserving first-seen order.
        """
        return tuple(merge_specs(self.state_reads(), self.state_writes()).values())

    @property
    def is_stochastic(self) -> bool:
        """Return whether this step samples an action and can contribute a score term.

        Returns:
            False for a deterministic step.
        """
        return False

    def __call__(self, state: BaseState, *, dt: float | jax.Array, key: jax.Array) -> BaseState:
        """Apply the step over ``dt``; return a state (see the class docstring for the contract).

        Quasistatic and discrete steps return the new state; dynamic steps return a sparse delta
        state via ``state.deltas(...)``. ``dt`` and ``key`` are keyword-only, so a step that ignores
        one or both still accepts them and the model calls every step the same way.

        Args:
            state: Live input `BaseState`.
            dt: Macro-step duration.
            key: JAX PRNG key; deterministic implementations may ignore it.

        Returns:
            A new state for a quasistatic or discrete step, or a sparse delta state for a dynamic
            step.

        Raises:
            NotImplementedError: Always, until a subclass supplies the step implementation.
        """
        raise NotImplementedError(f'{type(self).__name__} is {self.step_type}; implement __call__')


class StochasticStep(SimulationStep):
    """Mixin for a stochastic step: it records a trace, replays it, and scores it.

    A stochastic step samples a **trace** - the exogenous noise (a parameter-free draw, e.g. a
    standard-normal ``xi``) plus the realized action (e.g. the 0/1 ``divided`` outcome, or a
    displacement ``dx``) - and a single ``replay`` turns a trace into the step's effect. The forward
    ``__call__`` samples a trace and replays it pathwise; the top-level scoring drivers read a
    recorded trace back out of the post-step state and replay it again with parameters live, scoring
    the selected steps.

    The one per-instance knob is ``score_by_default``: whether this step contributes its ``logp`` when
    a scoring driver is called with no ``score=`` override. Whether the step is
    *reparameterizable* -
    and so whether an unscored replay can carry a pathwise gradient - is not a knob but intrinsic to
    the distribution and lives inside ``replay``: ``replay(..., pathwise=True)`` recomputes a
    reparameterizable value from the trace's exogenous noise with live parameters, while a discrete
    step has nothing to reparameterize and uses the realized action regardless. The model always
    passes ``pathwise = not scored``.

    Subclasses implement ``trace_writes`` (the recorded fields), ``sample_trace``,
    ``replay(state, trace, *, dt, pathwise)`` and ``logp(state, trace, dt)``;
    ``trace_from_state`` has a default (read the declared trace fields back out of a state) and is
    overridden only for a bespoke layout. ``_dist`` is the recommended single source of the
    distribution's parameters, shared by ``sample_trace`` and ``logp`` so a step cannot sample under
    one distribution and score under another.

    Attributes:
        score_by_default: Whether a scorer with ``score=None`` includes this stochastic step.
            Defaults to True.

    Methods:
        trace_writes: Declare ephemeral trace fields for one macro-step.
        sample_trace: Draw exogenous noise and realized discrete actions.
        trace_from_state: Recover the recorded trace from a post-step state.
        replay: Apply the effect of a recorded trace.
        logp: Score a recorded trace under the live state.
    """

    score_by_default: bool = eqx.field(static=True, default=True, kw_only=True)

    @property
    def is_stochastic(self) -> bool:
        """Return whether this step samples and scores an action.

        Returns:
            Always True for a stochastic step.
        """
        return True

    def state_requires(self) -> tuple:
        """Return read, physical-write, and ephemeral-trace specifications.

        Returns:
            A deduplicated tuple of `StateFieldSpec` objects.
        """
        return tuple(merge_specs(super().state_requires(), self.trace_writes()).values())

    def trace_writes(self) -> tuple:
        """Field specs for the recorded trace: ephemeral, reset to their defaults each macro-step.

        Trace fields are ordinary state fields (declared, allocated, co-emitted into the post-step
        state) but exist only to reconstruct and score *this* macro-step; the model resets them at
        each macro-step start. They are disjoint from ``state_writes`` - a stochastic output that must
        *persist* as physical state (read by a later macro-step) is a separate ``state_writes`` field
        written by ``replay``, never a trace field, which the reset would wipe. A **dynamic**
        (additive) trace field must default to the additive identity ``0`` so that reset-then-
        accumulate records exactly that step's own value.
        """
        return ()

    def _dist(self, state: BaseState, dt: float | jax.Array) -> object:
        """Distribution parameters conditioning on ``state`` - the single source for sample+score.

        Override and call from both ``sample_trace`` and ``logp`` so the sampler and the density
        never disagree. A step with a bespoke parameterization may route this differently but must
        keep them consistent.

        Args:
            state: Live pre-step state conditioning the distribution.
            dt: Macro-step duration.

        Returns:
            Distribution parameters in the subclass-defined layout.

        Raises:
            NotImplementedError: Always, until a subclass supplies the distribution.
        """
        raise NotImplementedError(f'{type(self).__name__} does not define _dist')

    def sample_trace(
        self, state: BaseState, *, dt: float | jax.Array, key: jax.Array
    ) -> dict[str, jax.Array]:
        """Draw the trace entries the pathwise replay consumes; return them (a dict of arrays).

        Must contain every entry ``replay(..., pathwise=True)`` reads: the exogenous noise, plus
        any directly-sampled (discrete) action. A *derived* realized value (a Brownian ``dx = mean
        + std * xi``) may be omitted - the pathwise replay recomputes it from the noise and records
        it, so computing it here too would only duplicate the reparameterization. Used only by the
        forward ``__call__``.

        Args:
            state: Live pre-step state conditioning the draw.
            dt: Macro-step duration.
            key: JAX PRNG key for the draw.

        Returns:
            A dictionary from trace-field names to their sampled arrays.

        Raises:
            NotImplementedError: Always, until a subclass supplies the sampler.
        """
        raise NotImplementedError(f'{type(self).__name__} must implement sample_trace')

    def trace_from_state(self, state: BaseState) -> dict[str, jax.Array]:
        """Read this step's recorded trace back out of a (post-step) state.

        The default reads every declared ``trace_writes`` field back by name; override only for a
        bespoke trace layout.

        Args:
            state: Post-step state containing the recorded trace.

        Returns:
            A dictionary from declared trace-field names to their arrays.
        """
        return {spec.name: state[spec.name] for spec in self.trace_writes()}

    def replay(
        self,
        state: BaseState,
        trace: dict[str, jax.Array],
        *,
        dt: float | jax.Array,
        pathwise: bool,
    ) -> BaseState:
        """Apply the step's effect from ``trace`` and write the trace fields - the single effect source.

        Returns the same shape as ``__call__`` (a new state for quasistatic/discrete, a sparse delta
        for dynamic). ``pathwise=True`` recomputes reparameterizable quantities from the trace's
        exogenous noise with live parameters (``dx = mean(state) + std(theta) * xi``); ``pathwise
        =False`` uses the realized value frozen. A step with nothing to reparameterize (a discrete
        node) uses the realized action regardless of ``pathwise``.

        Args:
            state: Live pre-step state.
            trace: Recorded exogenous noise and realized actions.
            dt: Macro-step duration.
            pathwise: Whether to recompute reparameterizable values with live parameters.

        Returns:
            A new state for a quasistatic or discrete step, or a sparse delta state for a dynamic
            step.

        Raises:
            NotImplementedError: Always, until a subclass supplies replay semantics.
        """
        raise NotImplementedError(f'{type(self).__name__} must implement replay')

    def logp(
        self, state: BaseState, trace: dict[str, jax.Array], dt: float | jax.Array
    ) -> jax.Array:
        """Log-density this step assigns to ``trace``'s action under ``_dist(state, dt)``.

        The conditioning ``state`` is live during replay; the caller has already
        ``stop_gradient``-ed the ``trace``, so only the recomputed distribution carries gradient.

        Args:
            state: Live pre-step state conditioning the distribution.
            trace: Detached recorded trace to score.
            dt: Macro-step duration.

        Returns:
            Scalar log-probability contribution for the trace.

        Raises:
            NotImplementedError: Always, until a subclass supplies the density.
        """
        raise NotImplementedError(f'{type(self).__name__} must implement logp')

    def __call__(self, state: BaseState, *, dt: float | jax.Array, key: jax.Array) -> BaseState:
        """Sample a trace and replay it pathwise as the forward sampler.

        Args:
            state: Live pre-step state.
            dt: Macro-step duration.
            key: JAX PRNG key for the trace draw.

        Returns:
            A new state or sparse dynamic delta, according to the step type.
        """
        trace = self.sample_trace(state, dt=dt, key=key)
        return self.replay(state, trace, dt=dt, pathwise=True)


def _accumulate_dynamic(state, deltas):
    """Sum sparse delta states, alive-mask the cell-scope totals, and apply them in one op.

    ``deltas`` is the list of sparse delta states returned by the dynamic steps (each evaluated at
    the same post-quasistatic ``state``). Fields are summed by name via attribute access (``None``
    means "not touched by this step" and is skipped), so multiple dynamic writers of a field
    accumulate. Cell-scope totals are zeroed on dead slots; the summed delta is then applied with
    ``eqx.apply_updates``, which leaves every untouched (``None``) field exactly as it was.
    """
    totals = {}
    for name, spec in state.specs.items():
        contribs = [getattr(d, name) for d in deltas if getattr(d, name) is not None]
        if not contribs:
            continue
        total = sum(contribs[1:], contribs[0])
        if spec.scope == 'cell':  # dead cells are not integrated; globals (grids) are not masked
            mask = state.alive.reshape((state.n_cells,) + (1,) * (total.ndim - 1))
            total = jnp.where(mask, total, jnp.zeros_like(total))
        totals[name] = total
    return eqx.apply_updates(state, state.deltas(**totals))


class Model(eqx.Module):
    """An ordered pipeline of steps with a validated field dataflow.

    ``Model`` is callable: ``model(state, *, dt, key)`` advances the state by one macro-step. The
    forward pass is a pure sampler. Top-level ``trajectory_logp`` and ``transition_logp`` score its
    recorded stochastic choices through private replay machinery. Sampling and replay share one
    macro-step skeleton (``_run``).

    Attributes:
        steps: Ordered tuple of `SimulationStep` instances.

    Methods:
        __call__: Advance one hybrid macro-step.
        state_requires: Return the full step-declared state schema.
    """

    steps: tuple

    def __init__(self, steps: Iterable[SimulationStep]) -> None:
        """Build a model from an ordered iterable of steps, validating the dataflow.

        Args:
            steps: Simulation steps in pipeline order.

        Raises:
            ValueError: If declared fields conflict or a stochastic trace is invalid.
        """
        self.steps = tuple(steps)
        self._validate()

    def _validate(self):
        # The model does not police reads: a field may be read but never written - fixed by the
        # initial condition or a constant that never changes - which is perfectly valid. Validation
        # is only the write-conflict policy plus, via state_requires(), that no two steps declare
        # the same field name with different specs. Trace fields never enter this check (they are in
        # trace_writes, not state_writes) - they are per-step namespaced and reset each macro-step.
        dynamic_fields: set[str] = set()
        firm_writers: dict[str, str] = {}  # quasistatic single-writer tracking
        for step in self.steps:
            if step.step_type is StepType.DISCRETE:
                continue  # discrete steps are resets, exempt from write-conflict checks
            cls = type(step).__name__
            for spec in step.state_writes():
                field = spec.name
                if step.step_type is StepType.DYNAMIC:
                    if field in firm_writers:
                        raise ValueError(
                            f'Field {field!r} is written by quasistatic {firm_writers[field]} '
                            f'and by dynamic {cls}: a field cannot be both'
                        )
                    dynamic_fields.add(field)
                else:  # QUASISTATIC
                    if field in firm_writers:
                        raise ValueError(
                            f'Field {field!r} is written by {firm_writers[field]} and again by '
                            f'{cls}; a quasistatic field admits exactly one writer'
                        )
                    if field in dynamic_fields:
                        raise ValueError(
                            f'Field {field!r} is dynamic but quasistatic {cls} also writes it'
                        )
                    firm_writers[field] = cls
        # Trace fields must be unique per stochastic step (namespaced) and must not shadow a physical
        # or base field. merge_specs silently dedups exactly-equal specs, so two steps declaring the
        # same trace field would share one allocation and clobber each other's recorded action; and a
        # trace field named like a real field would be wiped by the per-macro-step reset. Enforce both.
        # We deliberately do NOT reserve state_reads names: a downstream step reading an upstream
        # trace within the same macro-step (the Emit -> React pattern) is exactly a state_read that
        # equals a trace_write, so reserving reads would forbid that intended consumption path.
        reserved = {spec.name for spec in BASE_SPECS}
        for step in self.steps:
            reserved.update(spec.name for spec in step.state_writes())
        trace_owner: dict[str, int] = {}
        for i, step in enumerate(self.steps):
            if not step.is_stochastic:
                continue
            for spec in step.trace_writes():
                name = spec.name
                if name in trace_owner:
                    j = trace_owner[name]
                    raise ValueError(
                        f'Trace field {name!r} is declared by two stochastic steps (indices {j} '
                        f'{type(self.steps[j]).__name__} and {i} {type(step).__name__}); trace '
                        f'fields must be unique per step - namespace them (e.g. by an instance tag)'
                    )
                if name in reserved:
                    raise ValueError(
                        f'Trace field {name!r} of {type(step).__name__} collides with a base or '
                        f'physical (state_writes) field; a persistent field cannot double as a '
                        f'trace, since the per-macro-step reset would wipe it'
                    )
                if step.step_type is StepType.DYNAMIC and spec.default != 0:
                    raise ValueError(
                        f'Dynamic trace field {name!r} of {type(step).__name__} has default '
                        f'{spec.default!r}; a dynamic (additive) trace field must default to 0, or '
                        f'the per-macro-step reset-then-accumulate records default + value rather '
                        f'than the increment this step contributed'
                    )
                trace_owner[name] = i
        # Trigger the full consistency merge (cross-step AND against the base specs) in one line,
        # without owning a schema: the state constructor re-does this merge to allocate the state.
        merge_specs(BASE_SPECS, self.state_requires())

    def _reset_traces(self, state):
        """Reset every stochastic step's trace fields to their defaults (start of a macro-step).

        Ephemeral trace fields exist to reconstruct and score a single macro-step, so they must not
        leak across macro-steps. Resetting to the declared default before the phases run makes each
        recorded trace per-step; for a dynamic (additive) trace field the default is ``0``, so
        reset-then-accumulate records exactly that step's own value.

        Returns:
            State with every trace field reset to its declared default.
        """
        resets = {}
        for step in self.steps:
            if step.is_stochastic:
                for spec in step.trace_writes():
                    resets[spec.name] = jnp.full_like(state[spec.name], spec.default)
        return state.update(**resets) if resets else state

    def _run(self, state, per_step):
        """Drive one macro-step through the A/B/C phase skeleton with a per-step operation.

        ``per_step(index, step, live_state)`` returns the step's output (a new state for quasistatic/
        discrete, a sparse delta for dynamic) and may accumulate side data (log-probabilities) in its
        closure. Phase A (quasistatic) and phase C (discrete) run in pipeline order, each step seeing
        earlier updates; phase B (dynamic) evaluates every step at the *same* post-quasistatic state
        and applies the summed, alive-masked delta once. Trace fields are reset first.
        """
        state = self._reset_traces(state)

        # Phase A - quasistatic constraints, in pipeline order (each sees prior updates).
        for i, step in enumerate(self.steps):
            if step.step_type is StepType.QUASISTATIC:
                state = per_step(i, step, state)

        # Phase B - dynamic: evaluate every step at the SAME post-quasistatic state, then sum the
        # sparse deltas and apply them once. Each step baked its own dt-scaling in, so no scaling
        # happens here.
        deltas = [
            per_step(i, step, state)
            for i, step in enumerate(self.steps)
            if step.step_type is StepType.DYNAMIC
        ]
        if deltas:
            state = _accumulate_dynamic(state, deltas)

        # Phase C - discrete events, in pipeline order (each sees earlier jumps).
        for i, step in enumerate(self.steps):
            if step.step_type is StepType.DISCRETE:
                state = per_step(i, step, state)

        return state

    def __call__(self, state: BaseState, *, dt: float | jax.Array, key: jax.Array) -> BaseState:
        r"""Advance ``state`` by one macro-step of size ``dt``; return the new state.

        One macro-step is the first-order Lie-Trotter composition of three phase maps,

        $$s_{n+1} = \left(\Phi_\mathrm{discrete} \circ \Phi_\mathrm{dynamic} \circ \Phi_\mathrm{quasistatic}\right)(s_n),$$

        after which the simulation time ``t`` advances by ``dt``. Phase A - quasistatic: each
        quasistatic step's ``__call__`` runs in pipeline order, so a fast field is slaved to the
        current slow state and later steps see earlier updates. Phase B - dynamic: every dynamic step
        is evaluated at that single post-quasistatic state (so all derivatives see the *same* state,
        not a Gauss-Seidel sweep), each returns a sparse delta state, and the summed, alive-masked
        delta is applied in one op. Phase C - discrete: each discrete step's ``__call__`` runs in
        pipeline order (each sees earlier jumps).

        Args:
            state: Input `BaseState` at the current macro-step boundary.
            dt: Macro-step duration.
            key: JAX PRNG key, split once per model step.

        Returns:
            State at the next macro-step boundary with time advanced by ``dt``.
        """
        keys = jax.random.split(key, len(self.steps))

        def per_step(i, step, live_state):
            return step(live_state, dt=dt, key=keys[i])

        out = self._run(state, per_step)
        return out.set('t', out['t'] + dt)  # advance simulation time by dt

    def _replay_transition(self, live_state, recorded_next, dt, scored):
        """Replay one recorded macro-step and return its live next state and selected score.

        ``live_state`` is deliberately not detached: a trajectory driver carries it across numerical
        macro-step boundaries. ``recorded_next`` is detached structurally and supplies only the
        stochastic traces. Callers own any statistical conditioning boundary and resolve ``scored``
        once before driving one or more transitions.
        """
        recorded_next = jax.lax.stop_gradient(recorded_next)
        total = []  # trace-time list of per-step log-densities, summed at the end

        def per_step(i, step, state):
            if not step.is_stochastic:
                return step(state, dt=dt, key=_PLACEHOLDER_KEY)  # deterministic, params live
            trace = step.trace_from_state(recorded_next)
            if i in scored:
                total.append(step.logp(state, trace, dt))
                return step.replay(state, trace, dt=dt, pathwise=False)
            return step.replay(state, trace, dt=dt, pathwise=True)  # discrete replay ignores it

        replayed = self._run(live_state, per_step)
        replayed = replayed.set('t', replayed['t'] + dt)
        return replayed, sum(total, jnp.asarray(0.0))

    def _resolve_score(self, score):
        """Resolve a ``score=`` argument to a static ``frozenset`` of indices into ``self.steps``.

        ``None`` -> the ``score_by_default`` stochastic steps; ``'all'`` -> every stochastic step; an
        iterable of ints -> those indices. A boolean mask is rejected rather than reinterpreted
        (``bool`` subclasses ``int``, so a mask would silently read as the index set ``{0, 1}``);
        selecting a non-stochastic step is an error.
        """
        stochastic = frozenset(i for i, s in enumerate(self.steps) if s.is_stochastic)
        if score is None:
            return frozenset(i for i in stochastic if self.steps[i].score_by_default)
        if isinstance(score, str):
            if score == 'all':
                return stochastic
            raise ValueError(
                f'score={score!r} not understood; use None, "all", or an iterable of step indices'
            )
        seq = list(score)
        if any(_is_bool_like(x) for x in seq):
            raise ValueError(
                'score does not accept boolean masks; pass the indices of the steps to score'
            )
        chosen = frozenset(int(i) for i in seq)
        n = len(self.steps)
        out_of_range = sorted(i for i in chosen if i < 0 or i >= n)
        if out_of_range:
            raise ValueError(
                f'score selects out-of-range step indices {out_of_range} (model has {n} steps)'
            )
        non_stochastic = sorted(chosen - stochastic)
        if non_stochastic:
            raise ValueError(f'score selects non-stochastic step indices {non_stochastic}')
        return chosen

    def state_requires(self) -> tuple[StateFieldSpec, ...]:
        """The union of every step's required field specs (deduplicated tuple of specs).

        Deduplicates specs that are exactly equal and keeps every distinct field; raises when two
        steps declare the same field name with different properties (shape, dtype, scope, ...).
        The always-present base fields are the state's concern, not the model's;
        ``build_state_from_model`` combines them with this to form the full allocation schema.

        Returns:
            A deduplicated tuple of `StateFieldSpec` objects required by the model's steps.

        Raises:
            ValueError: If two steps declare the same field name with different properties.
        """
        return tuple(merge_specs(*(step.state_requires() for step in self.steps)).values())


def check_stochastic_step(
    step: StochasticStep,
    state: BaseState,
    *,
    dt: float | jax.Array = 1.0,
    key: jax.Array,
) -> dict[str, jax.Array]:
    """Assert a stochastic step's ``replay`` co-emits its trace, so scoring reads it back.

    The one contract a ``StochasticStep`` can silently violate is a ``replay`` that forgets to write
    a trace field into the state (or delta) it returns: the forward then records the reset default,
    ``trace_from_state`` reads that default back, and ``logp`` scores garbage - with no error. This
    helper drives the step and checks the round-trip in two ways:

    - **Co-emission (discrete and dynamic alike).** The step is driven twice from trace fields
      pre-filled with two *different* sentinels. A ``replay`` that co-emits a field derives it from
      the sampled trace and the physical state, so it records the *same* value both times; a field it
      forgets keeps the differing pre-fill - absent (``None``) in a dynamic delta, or the leaked
      sentinel in a discrete full state - so the two recorded values disagree and the check fires.
      (This is why resetting to the default and inspecting one run is not enough: a discrete
      ``replay`` that forgets a field leaves the default there, indistinguishable from writing it.)
    - **Fidelity.** Every entry ``sample_trace`` drew must survive into the recorded trace unchanged,
      so the exogenous noise the pathwise replay consumes is preserved. This assumes the documented
      convention that ``sample_trace`` returns each entry *as it will be recorded* (any eligibility
      masking happens inside ``sample_trace``, as in the ``MaybeDivide`` example) - a step that
      instead masks a draw inside ``replay`` should not list that entry from ``sample_trace``.

    Returns the recorded trace on success; raises ``AssertionError`` on a round-trip failure and
    ``TypeError`` / ``ValueError`` for a misused (non-stochastic or trace-less) step.

    Args:
        step: Stochastic step whose trace contract to validate.
        state: Pre-step state with allocated trace fields.
        dt: Macro-step duration. Defaults to 1.0.
        key: JAX PRNG key for the repeatable trace draw.

    Returns:
        Recorded trace dictionary on success.

    Raises:
        TypeError: If ``step`` is not stochastic.
        ValueError: If ``step`` declares no trace fields.
        AssertionError: If replay fails to co-emit or preserve a trace field.
    """
    if not step.is_stochastic:
        raise TypeError(f'{type(step).__name__} is not a stochastic step (nothing to round-trip)')
    specs = step.trace_writes()
    if not specs:
        raise ValueError(
            f'{type(step).__name__} declares no trace_writes; a stochastic step must record a trace'
        )

    def _record(fill):
        base = state.update(**{spec.name: jnp.full_like(state[spec.name], fill) for spec in specs})
        return step.trace_from_state(step(base, dt=dt, key=key))

    rec0 = _record(0.0)
    rec1 = _record(
        1.0
    )  # a distinct sentinel: a co-emitted field ignores it, a forgotten one leaks it
    for spec in specs:
        a, b = rec0.get(spec.name), rec1.get(spec.name)
        if a is None or b is None or not bool(jnp.allclose(a, b)):
            raise AssertionError(
                f'trace field {spec.name!r} of {type(step).__name__} is not co-emitted by replay; '
                f'replay must write every declared trace field into the state (or delta) it returns'
            )

    base0 = state.update(**{spec.name: jnp.full_like(state[spec.name], 0.0) for spec in specs})
    sampled = step.sample_trace(base0, dt=dt, key=key)  # same key as _record(0.0) -> same draw
    for name, value in sampled.items():
        rec = rec0.get(name)
        if rec is None or not bool(jnp.allclose(rec, value)):
            raise AssertionError(
                f'sampled trace field {name!r} of {type(step).__name__} did not round-trip through '
                f'replay/trace_from_state; the recorded value must match the sampled draw'
            )
    return rec0
