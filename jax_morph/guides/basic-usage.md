# Basic usage

A jax-morph simulation starts with an ordered `Model`, a state class synthesized from that model,
and an explicitly initialized state. This minimal model grows one cell toward a target radius.

```python
import jax.numpy as jnp
import jax_morph as jxm

model = jxm.Model([
    jxm.physics.SaturatingCellGrowth(max_radius=1.0),
])

State = jxm.build_state_from_model(model, name='growth')
empty = State.init_empty(capacity=8, n_space_dim=2, n_types=1)
state = empty.update(
    alive=empty.alive.at[0].set(True),
    radius=empty.radius.at[0].set(0.5),
    celltype=empty.celltype.at[0, 0].set(1.0),
    growth_rate=empty.growth_rate.at[0].set(0.4),
)

final = jxm.simulate(model, state, n_steps=20, dt=0.1)
print(final.radius[0], final.t)
```

`build_state_from_model` combines the always-present cell fields with every field declared by the
model's steps. `init_empty` allocates a fixed-capacity, zero-live-cell state; the following
`update` supplies this simulation's initial condition.

For a stochastic model, pass a JAX PRNG key to `simulate`. Set `history=True` when a complete
trajectory is needed for analysis, persistence, or `trajectory_logp` scoring.

The [Key Concepts](concepts.md) guide explains the three step types and macro-step
ordering. [Writing a Custom Step](extending.md) gives the complete extension contract.
