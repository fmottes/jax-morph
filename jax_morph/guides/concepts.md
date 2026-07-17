# Key concepts

## A simulation integrates a hybrid dynamical system

A `Model` defines a dynamical system and `simulate` integrates it forward to a physical end time. The
system is *hybrid*: some processes are fast constraints solved at the current configuration
(quasistatic), some evolve at a finite rate (dynamic), and some are instantaneous events (discrete).

One macro-step advances the state over a time increment `dt` by a Lie-Trotter operator split: the
quasistatic, dynamic, and discrete phases are applied as sub-maps composed in a fixed order into a
single macro-step operator,

$$F_{dt} = F^{\mathrm{disc}}_{dt} \circ F^{\mathrm{dyn}}_{dt} \circ F^{\mathrm{qs}}_{dt}, \qquad s_{n+1} = F_{dt}(s_n).$$

A whole simulation iterates this map, so it is the forward composition of `n_steps` identical
operators and simulated time is real physical time,

$$s(T) = F_{dt}^{\,n_{\mathrm{steps}}}(s_0), \qquad T = n_{\mathrm{steps}}\, dt.$$

The split is first-order accurate: its `O(dt)` error shrinks as `dt` is refined toward the same
physical `T`.

## Models are ordered step pipelines

A `Model` is an ordered collection of simulation steps. Each step declares the per-cell (and
global) state fields it reads and writes.

Steps belong to one of three kinds:

| Step type | Meaning | Return value |
|---|---|---|
| `QUASISTATIC` | Solve a fast constraint at the current slow state. | A complete updated state. |
| `DYNAMIC` | Contribute a finite-time increment. | A sparse delta created with `state.deltas(...)`. |
| `DISCRETE` | Apply an instantaneous event or reset. | A complete updated state. |

Within each macro-step, all quasistatic steps run first in pipeline order. Dynamic steps are then
evaluated from the same post-quasistatic state and their increments are summed. Discrete steps run
last in pipeline order. Simulation time advances by `dt` after these three phases.

## State comes from the model

`build_state_from_model(model)` returns a cached `BaseState` subclass containing the base fields
(`position`, `radius`, `celltype`, `alive`, and `t`) plus every field required by the steps. Arrays
have fixed capacity so compiled rollouts keep static shapes and unused cell slots have `alive == False`.

State updates are functional:

```python
state = state.set('radius', new_radius)
state = state.update(radius=new_radius, alive=new_alive)
delta = state.deltas(radius=radius_increment)
```

Cell-scope fields gain a leading capacity axis. Global fields, including simulation time, are not
indexed by cell.

## Physics and control compose through fields

Physics and control couple only through named state fields; there is no other channel between steps.
Each step reads the fields it needs and writes its outputs back, and a downstream step consumes those
outputs by name. A typical chain: diffusion writes a chemical signal field, a controller reads that
signal and writes a per-cell `growth_rate`, growth reads `growth_rate` and enlarges `radius`, and
division reads `radius` to decide which cells split. Because the wiring lives entirely in these
read/write declarations - validated against each other when the model is built - you recompose a
model by adding, removing, or reordering steps rather than editing a monolithic simulator.

## Differentiability

The forward pass of `simulate` is a pure sampler that is pathwise-differentiable end to end.

### Parameters: jax.Array vs Python typed

Which quantities are differentiable is a typing choice: a step parameter stored as a `jax.Array` is
*traced*, so it is optimizable and passed into `jit` as an argument, whereas a plain Python scalar is
*static* and baked into the compiled program as a constant.

**Store as a `jax.Array` anything you want to learn or vary without recompiling, and use a Python type (float, int, list, tuple, etc.) for anything fixed at compile time.**

**Only works if you use Equinox filtered JAX transformations (e.g. `eqx.filter_jit`, `eqx.filter_vmap`, `eqx.filter_grad`, etc.)**

### Pathwise

Continuous and reparameterized quantities keep a gradient through `simulate`. Differentiate an
objective of the final or trajectory state with respect to any traced parameter directly. This is the
default path and, for reparameterized noise, gives unbiased low-variance gradients. Put `simulate`
**inside** the differentiated objective so the sampled state remains on the gradient path.

### Score-based

Discrete choices such as division, death, and type flips are not reparameterizable. A
`StochasticStep` records its sampled action into an ephemeral trace field, which rides in the complete
history returned by `simulate(..., history=True)`. Pass that history to
`trajectory_logp(model, history, dt)` to obtain one log-density term per transition, the basis for a
score-function (REINFORCE) gradient. The recommended training boundary is the opposite of the
pathwise case: sample and detach complete histories **before** calling the differentiated score
surrogate, then differentiate only `trajectory_logp` with the candidate model live.

| Estimator | `simulate` placement | Differentiated path |
|---|---|---|
| Pathwise | Inside the objective | Through the final state or history |
| REINFORCE | Before the score surrogate | Through `trajectory_logp` on fixed recorded histories |

For a sampled discrete choice, use either its straight-through pathwise surrogate or its
score-function estimator, not both. See the [pathwise](optimization/pathwise.md) and
[REINFORCE](optimization/reinforce.md) guides for complete patterns.
