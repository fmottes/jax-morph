# Core abstractions

The core (`jax_morph.core`) defines the small set of generic abstractions that physics and control
build on. This page describes each abstraction and its purpose; the concrete physical steps that
instantiate them live outside the core (see the API reference). For how these pieces run together in
one macro-step, see [Key concepts](concepts.md); to implement your own step against these
abstractions, see [Writing custom steps](extending.md).

## State

The typed data a simulation carries. A state is not hand-written: it is *synthesized* from the fields
the model's steps declare, so every field is a real `jax.Array` attribute (`state.position`,
`state.custom_field`) allocated to a fixed capacity that keeps compiled rollouts statically shaped.

### StateFieldSpec

The declaration of one field: its name, shape, `scope` (`'cell'` per-cell or `'global'`), the default
that fills empty or reset slots, and whether a daughter inherits it on division. Steps pass
[`StateFieldSpec`][state] tuples around as their read/write contract; the always-present base fields
ship as prefilled specs (`POSITION`, `RADIUS`, `CELLTYPE`, `ALIVE`, `TIME`).

### BaseState

The generated state PyTree. `build_state_from_model(model)` returns and caches a [`BaseState`][state]
subclass carrying the base fields plus every field the steps require. Updates are functional: `set`
(one field), `update` (several fields), and `deltas` (a sparse dynamic-step increment).

## Step

The atomic unit of dynamics. A step is an Equinox module that declares the fields it reads and writes
and implements one method, `__call__(state, *, dt, key) -> state`. Steps know nothing about one
another; reconciling their declarations is the model's job.

### StepType

The kind of dynamics a step contributes, which fixes how its return value is interpreted:
[`QUASISTATIC`][step] (a fast constraint solved at the current state), `DYNAMIC` (a finite-rate
increment over `dt`, returned as a sparse delta), or `DISCRETE` (an instantaneous event or reset).

### SimulationStep

The deterministic step contract: [`SimulationStep`][step] declares its dataflow through
`state_reads()` / `state_writes()` and applies itself in `__call__` according to its `step_type`.

### StochasticStep

A step that additionally samples and scores an action. A [`StochasticStep`][step] records a **trace**
(the exogenous noise plus the realized action) into ephemeral trace fields, and a single `replay`
turns a trace into the step's effect. The forward pass samples then replays; the scoring drivers
replay recorded traces with parameters live. That split is what lets one step be both a sampler and a
scorable density (`logp`).

## Model

An ordered pipeline of steps with a *validated* field dataflow. Building a [`Model`][model] checks the
steps' declarations for conflicts and synthesizes the exact state type they need. Calling it,
`model(state, *, dt, key)`, advances one hybrid macro-step (the quasistatic -> dynamic -> discrete
split). The forward pass is a pure sampler; scoring is a separate, private replay path.

## Simulation drivers

The top-level functions that run a model across many macro-steps.

### simulate

[`simulate`][simulate] rolls a model out for `n_steps` of size `dt`, returning the final state or the
complete history. The forward pass is a pure sampler and is pathwise-differentiable end to end.

### trajectory_logp and transition_logp

The scoring drivers for recorded stochastic choices. [`trajectory_logp`][scoring] replays a full
history while carrying a live reconstructed state across macro-steps; `transition_logp` scores one
transition against an explicitly observed conditioning state. Both return per-choice log-density
terms for a score-function gradient.

## Space

The injectable boundary conditions, stored on the state as a `displacement` / `shift` pair and read
by physics steps. [`free_space`][geometry] (unbounded) and `periodic(box)` are provided, and
`neighbor_sum` is a dense pairwise-reduction combinator for building custom pairwise layers. Whether
a given step is valid in a given space is the user's responsibility, not enforced.

## Autodiff primitives

The differentiability toolkit steps reach for: numerically-safe ops ([`safe_norm`][ad-utils],
`safe_log`, `safe_divide`), forward-exact / backward-smooth **straight-through** estimators for
discrete operations (`heaviside_st`, `argmax_st`, the `sample_*_st` samplers), and log-density
helpers (`bernoulli_logp`, `categorical_logp`, `gaussian_logp`). The invariant: a straight-through
function's forward output is bit-identical to the exact op, and only its backward pass uses the
smooth surrogate.

## Serialization

Versioned, non-executable save/load for models, states, and trajectories. The format pairs JSON
metadata with NPY numeric payloads and never uses pickle or runs artifact-supplied code.
[`TrajectoryRecord`][serialization] bundles a stacked history with the metadata needed to reload it.

[state]: api/core/state.md
[step]: api/core/step-primitives.md
[model]: api/core/model.md
[simulate]: api/core/simulate.md
[scoring]: api/core/scoring.md
[geometry]: api/core/geometry.md
[ad-utils]: api/core/utilities.md
[serialization]: api/serialization.md
