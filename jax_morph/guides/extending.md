# Writing a custom step

This is the reference for implementing new simulation steps. A **step** is the unit of
composition in jax-morph: a small `equinox.Module` that reads some state fields, writes others, and
declares what *kind* of dynamics it contributes. A `Model` is an ordered pipeline of steps; the
library validates their dataflow, generates a typed state for them, and runs them with a hybrid
integrator. Everything a physics or control layer does is a step.

If you only want to read the finished contract, jump to [the checklist](#checklist). Otherwise this
page builds it up: the mental model, the building blocks, the three deterministic step types, then
the stochastic (scorable) contract, and finally how steps compose, run, and are tested.

---

## Mental model

Every step implements one uniform method:

```python
def __call__(self, state, *, dt, key) -> state
```

`dt` (the macro-step size) and `key` (a JAX PRNG key) are **keyword-only**, so a step that ignores
one or both still accepts them and the model can call every step the same way. What the returned
value *means* is decided by the step's `step_type`, not by the method:

| `step_type`     | Returns                                  | Runs                                              |
|-----------------|------------------------------------------|---------------------------------------------------|
| `QUASISTATIC`   | the **new state** (a constraint solve)   | in pipeline order; each step sees prior updates   |
| `DYNAMIC`       | a **sparse delta** (`dt`-scaled)         | all evaluated at the *same* state; deltas summed  |
| `DISCRETE`      | the **new state** (a jump)               | in pipeline order; each step sees earlier jumps   |

One macro-step is a Lie-Trotter split into three phases, in this fixed order:

1. **A - quasistatic.** Each quasistatic step runs in pipeline order. A fast field is slaved to the
   current slow state, and a later step sees the earlier steps' updates.
2. **B - dynamic.** Every dynamic step is evaluated at the *single* post-quasistatic state (so all
   rates see the same state, not a Gauss-Seidel sweep). Each returns its own sparse increment; the
   model sums them across writers, zeroes cell-scope totals on dead slots, and applies the total in
   one op.
3. **C - discrete.** Each discrete step runs in pipeline order (each sees earlier jumps).

Then simulation time `t` advances by `dt`. `simulate(model, state, n_steps, dt, key)` scans this.

Scoring is a **separate** pass. The forward pass is a pure sampler;
`trajectory_logp(model, history, dt)` re-runs the same three phases to score the stochastic choices
while carrying the reconstructed state across macro-steps (see
[Stochastic steps](#stochastic-steps)). A deterministic step needs nothing extra for scoring - it is
just re-executed with parameters live. Use `transition_logp` when a single intermediate state is an
explicitly observed conditioning boundary.

---

## Building blocks

### `StateFieldSpec` - declaring a field

A step declares its inputs and outputs as tuples of `StateFieldSpec`. A spec is self-describing (it
carries its own `name`), immutable, and calling it returns a customised copy:

```python
import jax_morph as jxm

# a new per-cell scalar field
SIGNAL = jxm.StateFieldSpec('signal', shape=(), heritable=False)

# a customised copy of a base spec (see below)
POS_ENV = jxm.POSITION(heritable=False)
```

The attributes:

| Attribute   | Meaning                                                                              |
|-------------|--------------------------------------------------------------------------------------|
| `name`      | The field name; becomes a real attribute on the generated state (`state.signal`).    |
| `shape`     | The **trailing** (per-cell) shape. A `'cell'`-scope field gets a leading capacity axis in front of this automatically. |
| `dtype`     | Array dtype; `None` means the JAX default float.                                     |
| `heritable` | Cell scope only: does a daughter inherit the mother's value on division? `True` for a property of the cell (size, type, internal state); `False` for a transient/contextual quantity (a recorded action, an environment reading), which starts a daughter at `default`. |
| `default`   | Fill value: allocation of empty states, reset of ephemeral trace fields, and newborn non-heritable fields. |
| `scope`     | `'cell'` (per-cell: leading capacity axis, alive-masked, resized on division) or `'global'` (not indexed by cell: time, a scalar, an Eulerian grid array). |

The always-present **base fields** have prefilled specs you reuse directly: `jxm.POSITION`,
`jxm.RADIUS`, `jxm.CELLTYPE`, `jxm.ALIVE`, `jxm.TIME`. The generated state constructor owns the
concrete shapes of the dimension-dependent ones (`position -> (n_space_dim,)`,
`celltype -> (n_types,)`), so you never hardcode those.

### The state object

You never write the state class; `build_state_from_model(model)` generates it from the union of the
base fields and every step's `state_requires()`. Inside a step you use:

- `state.position`, `state.signal`, ... - fields as real attributes (also `state['signal']`).
- `state.set('signal', value)` - a functional single-field update (returns a new state).
- `state.update(signal=..., radius=...)` - a functional multi-field update.
- `state.deltas(radius=increment)` - a **sparse delta state**: only the named fields set, every
  other field `None` (the dynamic-step return; see below).
- `state.alive` (a boolean mask), `state.n_cells` (allocated capacity), `state.n_space_dim`.
- `state.displacement(Ra, Rb)` / `state.shift(R, dR)` - the injected boundary metric (free space or
  periodic), so geometry code works under either without change.

Capacity is fixed (arrays are statically shaped for `jit`/`vmap`/`scan`); dead slots carry
`alive == False`. Cell-scope work must therefore respect `alive` (see the masking notes per step
type).

---

## Deterministic steps

Subclass `SimulationStep`, set `step_type`, declare reads/writes, implement `__call__`.

### Parameter convention (important)

Store numeric parameters as **plain Equinox fields**. A Python scalar is static (baked into the
trace); a `jax.Array` is traced and therefore **optimizable** by `eqx.filter_grad`. Reserve
`eqx.field(static=True)` for values that determine structure - shape-determining ints, scan lengths,
namespacing tags, callables:

```python
class Grow(jxm.SimulationStep):
    step_type = jxm.StepType.DYNAMIC
    rate: jax.Array  # a jax.Array -> a trainable parameter
    n_space_dim: int = eqx.field(static=True, default=2)  # structural -> static
```

Pass `rate=jnp.array(0.5)` to make it optimizable, or `rate=0.5` to freeze it. All `jit`/`grad`
goes through `eqx.filter_*`, which splits traced arrays from static leaves for you.

### Quasistatic step

Returns the **new state**. Runs in pipeline order, so it may depend on fields an earlier quasistatic
step wrote this macro-step. A quasistatic field admits **exactly one** writer (validated). Because
it returns a full state, it owns its own alive-masking for cell-scope fields.

```python
import jax.numpy as jnp
import jax_morph as jxm


class Sense(jxm.SimulationStep):
    """Write a per-cell ``signal`` = distance from the live-cluster centroid."""

    step_type = jxm.StepType.QUASISTATIC

    def state_reads(self):
        return (jxm.POSITION, jxm.RADIUS, jxm.ALIVE)

    def state_writes(self):
        return (jxm.StateFieldSpec('signal', shape=(), heritable=False),)

    def __call__(self, state, *, dt, key):
        alive = state.alive.astype(state.radius.dtype)
        n_alive = jnp.clip(alive.sum(), 1.0, None)
        com = (state.position * alive[:, None]).sum(0) / n_alive
        dist = jxm.ad_utils.safe_norm(state.position - com, axis=-1)
        return state.set('signal', dist * alive)  # dead slots -> 0
```

### Dynamic step

Returns a **sparse delta** via `state.deltas(...)`. The written fields hold that field's increment
**over `dt`** - the step bakes in its own `dt`-scaling. Every dynamic writer is evaluated at the
same post-quasistatic state, and the model **sums** the deltas, so a dynamic step returns only its
own contribution, never a full state. Multiple dynamic writers of one field is legal and they
accumulate (this is how forces superpose). The model zeroes cell-scope totals on dead slots, so a
dynamic step does **not** need to mask dead cells itself.

```python
class Grow(jxm.SimulationStep):
    """Grow every radius at a constant rate."""

    step_type = jxm.StepType.DYNAMIC
    rate: jax.Array

    def state_writes(self):
        return (jxm.RADIUS,)

    def __call__(self, state, *, dt, key):
        return state.deltas(radius=self.rate * dt * jnp.ones_like(state.radius))
```

A dynamic step that needs pairwise geometry uses the `geometry` helpers, which keep it short and
boundary-agnostic:

```python
class Repel(jxm.SimulationStep):
    """A toy pairwise repulsion delta on position."""

    step_type = jxm.StepType.DYNAMIC
    strength: jax.Array

    def state_reads(self):
        return (jxm.POSITION, jxm.ALIVE)

    def state_writes(self):
        return (jxm.POSITION,)

    def __call__(self, state, *, dt, key):
        disp = jxm.geometry.pairwise_displacements(state.position, state.displacement)  # (N, N, d)
        dist = jxm.ad_utils.safe_norm(disp, axis=-1, keepdims=True)  # (N, N, 1)
        force = jxm.ad_utils.safe_divide(disp, dist**3)  # 1/r^2 along the separation
        total = self.strength * jxm.geometry.neighbor_sum(force, state.alive)  # (N, d)
        return state.deltas(position=total * dt)
```

### Discrete step

Returns the **new state** (a jump: division, death, a reset). Runs in pipeline order. Discrete
steps are **exempt** from the write-conflict checks (they are resets - last writer wins by design),
so two discrete steps may both touch `alive`. As with quasistatic steps, a discrete step owns its
own alive-masking.

A discrete step that samples randomness (division, stochastic death) is almost always a
**stochastic step** - it should be scorable. That is the next section.

---

## Stochastic steps

A stochastic step samples an action and can be **scored**: the top-level scoring drivers replay the
model and ask each selected stochastic step for the log-density of the action it took. This is what
makes the model a differentiable policy (REINFORCE) and what makes observed trajectories or
transitions scorable (MLE).

Subclass `StochasticStep` (which is itself a `SimulationStep`). You do **not** write `__call__` -
it is derived. You implement the trace contract.

### The trace

The **trace** is the recorded stochastic information a step needs to (a) reconstruct its effect and
(b) score its action. It carries two things:

- the **exogenous noise** - a parameter-free draw (a standard-normal `xi`, a uniform); and
- the **realized action** - what actually happened (the 0/1 `divided`, the displacement `dx`).

The noise lets a *reparameterized* step be replayed with **live** parameters (recompute
`dx = mean(theta) + std(theta) * xi` from the frozen `xi`); the realized action lets any step be
replayed **frozen** and lets `logp` score the value actually taken. Trace entries are ordinary state
fields (declared via `trace_writes()`, allocated, co-emitted into the post-step state) so they ride
the trajectory automatically and `history=True` records them.

Trace fields are **ephemeral**: the model resets them to their `default` at the start of every
macro-step. They exist only to reconstruct and score *this* macro-step. Two consequences:

- A stochastic output that must **persist** (be read by a *later* macro-step) is a separate
  `state_writes()` field written by `replay`, **never** a trace field - the reset would wipe it. A
  field read by a *later step in the same macro-step* is fine as a trace field (it is written and
  read before the next reset).
- A **dynamic** (additive) trace field must default to `0` (the additive identity), so
  reset-then-accumulate records exactly this step's own increment rather than a running sum.

### The methods

| Method                                   | You implement it? | Role                                                                 |
|------------------------------------------|-------------------|----------------------------------------------------------------------|
| `trace_writes(self)`                     | yes               | the trace field specs (ephemeral)                                    |
| `_dist(self, state, dt)`                 | recommended       | the distribution parameters - **one source** for sample and score    |
| `sample_trace(self, state, *, dt, key)`  | yes               | draw the trace entries the pathwise replay consumes                  |
| `replay(self, state, trace, *, dt, pathwise)` | yes          | the **single** source of the step's effect; also co-emits the trace  |
| `logp(self, state, trace, dt)`           | yes               | log-density of the trace's action under `_dist(state, dt)`           |
| `trace_from_state(self, state)`          | default provided  | read the trace back out of a state (override only for bespoke layout)|
| `__call__`                               | **no** (derived)  | `sample_trace` then `replay(pathwise=True)`                          |

The one per-instance knob is `score_by_default` (a static, keyword-only bool, default `True`):
whether this step contributes its `logp` when a scoring driver is called with no `score=` override.
Whether the step is *reparameterizable* is **not** a knob - it is intrinsic to the distribution and
lives inside `replay`. The model always passes `pathwise = not scored`; a reparameterizable
`replay` honours `pathwise=True` (recompute from the noise, live), a discrete `replay` ignores it
(nothing to reparameterize).

### Density helpers

`jxm.ad_utils` pairs each straight-through sampler with its log-density so a step stays a one-liner:

- `sample_bernoulli_st(key, p)` with `bernoulli_logp(outcome, p)` (boundary-safe at `p in {0, 1}`).
- `sample_categorical_st(key, logits)` with `categorical_logp(onehot, logits)`.
- a reparameterized Gaussian draw `x = mean + std * xi` with `gaussian_logp(x, mean, std)` (the
  density a Brownian / Langevin step scores; `std` is the deviation, assumed positive).

The samplers are **forward-exact, backward-smooth**: the forward value is a real sample, the
gradient flows through a smooth surrogate (so a sampled forward pass is still pathwise-differentiable
where that makes sense). Also available: `safe_norm`, `safe_log`, `safe_divide`, `straight_through`,
`heaviside_st`, `argmax_st`.

### Example A - a discrete Bernoulli step

Each live cell divides with probability `p`. The trace is the 0/1 action plus an **eligibility
mask** (who was alive to decide) - recording eligibility as data, rather than re-reading `alive` at
score time, keeps the score correct even though a real division mutates `alive` mid-macro-step.

```python
class MaybeDivide(jxm.StochasticStep):
    step_type = jxm.StepType.DISCRETE
    p: jax.Array  # a jax.Array -> the optimizable policy parameter

    def trace_writes(self):
        return (
            jxm.StateFieldSpec('divided', shape=(), heritable=False),
            jxm.StateFieldSpec('divide_eligible', shape=(), heritable=False),
        )

    def _dist(self, state, dt):
        return self.p * jnp.ones_like(state.radius)

    def sample_trace(self, state, *, dt, key):
        divided = jxm.ad_utils.sample_bernoulli_st(key, self._dist(state, dt))
        eligible = state.alive.astype(divided.dtype)
        return {'divided': divided * eligible, 'divide_eligible': eligible}

    def replay(self, state, trace, *, dt, pathwise):
        # discrete: nothing to reparameterize, pathwise is ignored; write the recorded action
        return state.update(divided=trace['divided'], divide_eligible=trace['divide_eligible'])

    def logp(self, state, trace, dt):
        prob = self._dist(state, dt)  # recomputed live through p during trace replay
        return jnp.sum(
            jxm.ad_utils.bernoulli_logp(trace['divided'], prob) * trace['divide_eligible']
        )
```

`trace_from_state` is not overridden - the default reads `divided` and `divide_eligible` back by
name. A real `Division` would do far more in `replay` (place daughters, copy heritable fields, mark
`alive`, bump an overflow counter) - all keyed off `trace['divided']`, so the re-run reproduces the
exact same event.

> **Naming convention - eligibility masks.** When a stochastic step's action applies only to a
> subset of cells (those alive or otherwise *eligible* to draw at decision time), record that subset
> as a trace field named **`{action}_eligible`**, where `{action}` is the step's action name or verb
> stem. In this codebase: action `divided` -> `divide_eligible`, `flipped` -> `flip_eligible`,
> `emit` -> `emit_eligible`, `react` -> `react_eligible`. `logp` multiplies the per-cell density by
> this mask so only the eligible cells are scored, and prefixing by the action keeps the field unique
> across steps (trace fields must be unique per model). Prefer this over a bare `eligible` and over
> re-reading `state.alive` at score time, which a mid-macro-step `alive` mutation (a real division)
> would make wrong.

### Example B - a reparameterized dynamic step

Kick a scalar field `u` by `dx = mean + std * xi`, with `std = exp(log_std)`. This is the pattern a
Brownian/Langevin step follows. Note `sample_trace` returns **only the noise**; `replay` derives
`dx` and records both - so the recorded `dx` is by construction the one `replay` computed (no "two
formulas that must agree"). The trace fields are additive dynamic fields defaulting to `0`, and
`replay` returns a **delta** (this is a dynamic step).

```python
class Kick(jxm.StochasticStep):
    step_type = jxm.StepType.DYNAMIC
    log_std: jax.Array
    tag: str = eqx.field(static=True, default='kick')  # namespaces the trace fields

    @property
    def _xi(self):
        return f'{self.tag}_xi'

    @property
    def _dx(self):
        return f'{self.tag}_dx'

    def state_writes(self):
        return (jxm.StateFieldSpec('u', shape=(), heritable=False),)

    def trace_writes(self):
        return (
            jxm.StateFieldSpec(self._xi, shape=(), heritable=False),  # noise, default 0
            jxm.StateFieldSpec(self._dx, shape=(), heritable=False),  # realized dx, default 0
        )

    def _dist(self, state, dt):
        mean = jnp.zeros_like(state.radius)
        std = jnp.exp(self.log_std) * jnp.ones_like(state.radius)
        return mean, std

    def sample_trace(self, state, *, dt, key):
        return {self._xi: jax.random.normal(key, state.radius.shape)}  # noise only

    def replay(self, state, trace, *, dt, pathwise):
        mean, std = self._dist(state, dt)
        if pathwise:
            dx = mean + std * trace[self._xi]  # recompute from noise with LIVE log_std
        else:
            dx = trace[self._dx]  # frozen realized displacement
        return state.deltas(**{'u': dx, self._xi: trace[self._xi], self._dx: dx})

    def logp(self, state, trace, dt):
        mean, std = self._dist(state, dt)
        z = (trace[self._dx] - mean) / std
        alive = state.alive.astype(std.dtype)
        return jnp.sum((-0.5 * z * z - jnp.log(std) - 0.5 * jnp.log(2.0 * jnp.pi)) * alive)
```

### Reparameterized vs frozen, and how gradients flow

The scoring drivers detach every recorded trace, then for each stochastic step pass
`pathwise = not scored`:

- **Scored** step -> `pathwise=False`. It accumulates `logp` and replays the **frozen** realized
  action. Its parameters get gradient through *its own* `logp` term (the score-function / REINFORCE
  gradient). Freezing the action is what prevents a straight-through double-count when a downstream
  step reads it.
- **Unscored** step -> `pathwise=True`. No `logp`. A reparameterizable step recomputes its value
  from the noise with live parameters, so a parameter's gradient **flows through** into a later
  *scored* step's `logp` (the within-step pathwise gradient). A discrete unscored step has nothing
  to reparameterize; its gradient dies there, which is unavoidable and correct.

`trajectory_logp` detaches the complete history once: frame zero becomes the initial state, and
each later frame supplies the next recorded trace. It carries the replayed state live across
numerical macro-step boundaries. A later score can therefore reach deterministic or unscored
reparameterized computations in earlier macro-steps. The actual gradient barriers are event-level:
a scored realized action is frozen, while its conditional physical effect may remain differentiable
with respect to incoming continuous state and parameters. `transition_logp` instead detaches its
observed state deliberately, making one statistical conditioning boundary. A displacement you want
treated as fixed data is a **deterministic** step, not an unscored reparameterized one.

---

## Composing a model

```python
model = jxm.Model([Sense(), Grow(rate=jnp.array(0.5)), MaybeDivide(p=jnp.array(0.3))])
```

`Model(steps)` validates the dataflow at construction. The rules:

- **Reads are not policed.** A field may be read but never written - fixed by the initial condition
  or a constant - which is valid. (A feedback loop where a step reads a field a later step writes is
  also valid; it carries the previous macro-step's value.)
- **One quasistatic writer per field.** Two quasistatic writers of the same field is an error; so is
  a field written by both a quasistatic and a dynamic step.
- **Dynamic writers accumulate.** Any number of dynamic steps may write the same field.
- **Discrete steps are exempt** from write-conflict checks (they are resets).
- **Same field name, same spec.** Two steps declaring the same field with *different* properties
  (shape, dtype, scope, ...) is an error; identical specs merge.
- **Trace fields are unique and non-shadowing.** Two stochastic steps must not declare the same
  trace field name (namespace them with a `tag`), and a trace field must not shadow a base or
  `state_writes` field (the per-macro-step reset would wipe a real field). This is why `Kick` above
  namespaces `xi`/`dx` by `tag`, and why two bare `MaybeDivide` instances in one model are rejected.

The order of `steps` is the pipeline order **within** each phase; the A/B/C phase order is fixed
regardless of how you interleave types in the list.

---

## Running and scoring

```python
import jax
import equinox as eqx

key = jax.random.PRNGKey(0)

# generate the typed state class and an initial condition
State = jxm.build_state_from_model(model)
s0 = State.init_empty(capacity=64, n_space_dim=2, n_types=1)
s0 = s0.update(alive=s0.alive.at[:8].set(True), radius=s0.radius.at[:8].set(0.5))

# forward (pure sampler). A model with any stochastic step REQUIRES a key.
s1 = model(s0, dt=1.0, key=key)  # one macro-step
traj = jxm.simulate(model, s0, n_steps=20, dt=1.0, key=key, history=True)

# score a rollout jointly. The result is one term per macro-step, not a scalar.
terms = jxm.trajectory_logp(model, traj, 1.0)
joint_lp = terms.sum()
grad = eqx.filter_grad(lambda m: jxm.trajectory_logp(m, traj, 1.0).sum())(model)

# score one transition only when s0 is intentionally an observed conditioning state.
conditional_lp = jxm.transition_logp(model, s0, s1, 1.0)
```

Choosing which steps contribute their `logp` (`score=`):

- `None` (default) -> every step whose `score_by_default` is `True`.
- `'all'` -> every stochastic step.
- an iterable of **indices** into `model.steps` -> exactly those (e.g. `score=[2]`). A boolean mask
  is rejected, not reinterpreted; selecting a non-stochastic or out-of-range index is an error.

With `history=True` the trajectory is the complete state sequence `s_0 .. s_n`; pass it directly to
`trajectory_logp`. Event traces are carried by destination frames one onward, while frame zero is
the initial condition. The returned array has shape `(n_steps,)`. Call `.sum()` for the joint
log-probability; for shaped returns explicitly reduce, for example
`-jnp.sum(stop_gradient(returns) * terms)`. Independent trajectories can be batched with an outer
`vmap`, but time is replayed sequentially because the reconstructed state is the live scan carry.
For MLE or importance ratios over a complete trace trajectory, `score='all'` is normally the right
selection. Use `transition_logp` for an explicitly conditioned observed transition.

---

## Testing your step

A `replay` that forgets to co-emit its trace fields fails **silently**: the forward records the
reset defaults, `trace_from_state` reads zeros back, and `logp` scores garbage. Guard it with the
shipped `jxm.check_stochastic_step(step, state, *, dt=1.0, key)`, which drives the step and asserts
the recorded trace round-trips (it catches a forgotten trace field for discrete and dynamic steps
alike, and raises `AssertionError` if the round-trip fails). Then check the gradient you expect:

```python
import numpy as np


def test_trace_round_trips():
    step = MaybeDivide(p=jnp.array(0.3))
    s0 = seed(...)  # a state with some live cells
    jxm.check_stochastic_step(step, s0, key=key)  # raises if replay drops a trace field


def test_score_gradient_reaches_the_param():
    m = jxm.Model([MaybeDivide(p=jnp.array(0.3))])
    s0 = seed(...)
    s1 = m(s0, dt=1.0, key=key)
    g = eqx.filter_grad(lambda mm: jxm.transition_logp(mm, s0, s1, 1.0))(m)
    assert float(g.steps[0].p) != 0.0  # the recorded action is scored through p
```

Useful checks, by step kind:

- **Deterministic:** a downstream `logp` gradient reaches an upstream deterministic parameter
  through the live re-run (the "wide" / MLE gradient); a scorer that read a detached recorded field
  would give zero.
- **Reparameterized dynamic:** `grad` w.r.t. the noise-scale parameter is nonzero through an
  *unscored* replay into a scored downstream step, and zero (through that path) when the step is
  instead scored/frozen. And the forward `simulate` stays pathwise-differentiable through it.
- **Trace reset:** over a multi-step rollout an additive dynamic trace field holds each step's own
  increment, not a running sum.

See `tests/core/test_logp.py` for the full worked suite (these toy steps live there).

---

## Checklist

Implementing a step:

- [ ] Subclass `SimulationStep` (deterministic) or `StochasticStep` (scorable).
- [ ] Set the `step_type` class var (`QUASISTATIC` / `DYNAMIC` / `DISCRETE`).
- [ ] Declare `state_reads()` and `state_writes()` as tuples of `StateFieldSpec`; reuse the base
      specs (`jxm.POSITION`, ...) for base fields.
- [ ] Numeric parameters are **plain fields** (`jax.Array` to optimize, Python scalar to freeze);
      `eqx.field(static=True)` only for structural values (shape ints, tags, callables).
- [ ] **Deterministic:** implement `__call__`. Quasistatic/discrete return the new state (mask
      `alive` yourself); dynamic returns `state.deltas(...)` with `dt`-scaling baked in.
- [ ] **Stochastic:** implement `trace_writes`, `_dist`, `sample_trace`, `replay(..., pathwise)`,
      `logp`; do **not** write `__call__`. Make `replay` the single source of the effect and have it
      co-emit the trace. Set `score_by_default` if it should not be scored by default.
- [ ] Trace fields ephemeral and non-shadowing; additive (dynamic) trace fields default to `0`;
      anything that must persist is a `state_writes` field, not a trace field.
- [ ] Add the step to a `Model` and confirm it constructs (dataflow validation) and its round-trip /
      gradient tests pass.

Conventions (enforced by `ruff` on `jax_morph/`; see `AGENTS.md`): `jax.Array` type hints (never
`np.ndarray`/`jnp.ndarray`), single quotes, ASCII only (write `epsilon`, `sqrt`, `->`; no Greek or
math symbols), Google-style docstrings, and no plan/issue labels in code.
