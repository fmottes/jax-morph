# Pathwise training

Use pathwise differentiation when the objective is a differentiable readout of the simulated state.
Here `simulate` **does belong inside** the differentiated loss: the gradient must flow through the
sampled state to the model parameters or initial state.

```python
import equinox as eqx
import jax.numpy as jnp
import jax_morph as jxm


def pathwise_loss(model, state, key):
    final = jxm.simulate(model, state, n_steps=100, dt=0.1, key=key)
    alive = final.alive.astype(final.position.dtype)
    n_alive = jnp.maximum(alive.sum(), 1)
    centre = (final.position * alive[:, None]).sum(0) / n_alive
    squared_radius = jnp.sum((final.position - centre) ** 2, axis=-1)
    gyration = jnp.sqrt((squared_radius * alive).sum() / n_alive)
    return jnp.square(gyration - 5.0)


loss_value, grads = eqx.filter_value_and_grad(pathwise_loss)(model, state, key)
```

Do not detach the final state or objective. A fixed `key` within one value-and-gradient evaluation
fixes that Monte Carlo sample; split fresh keys between optimizer updates if the model is stochastic.
Use `history=True` only when the objective reads intermediate frames, and use `checkpoint=True` when
trading extra recomputation for lower reverse-mode memory is worthwhile.

Numeric parameters stored as `jax.Array` leaves are traced and optimizable. Python scalar parameters
are static. Use `eqx.filter_*` transformations so these two categories are handled correctly. If the
initial state is the optimization variable, use an explicit wrapper whose first argument is the
state; Equinox filtered gradient transforms differentiate their first argument.

## What gradient does this estimate?

Gradients through deterministic continuous physics and reparameterized noise are pathwise. They are
the natural low-variance estimator for a smooth state readout.

Straight-through discrete events such as `Division` can also pass a useful surrogate derivative, but
that derivative is biased through discrete structure such as whether a slot became alive. A smooth
formula applied after a hard event does not make the event itself differentiable.

When the objective is dominated by a hard sampled outcome such as cell count, use the
[REINFORCE sample-then-score pattern](reinforce.md). For any one sampled discrete choice, choose its
straight-through pathwise surrogate or its score-function estimator; do not add both gradients for
that choice.

## Batch stochastic objectives

For reparameterized stochastic dynamics, average several pathwise objectives and differentiate the
average. Sampling remains inside the differentiated function because each final state carries its
own pathwise gradient:

```python
def batch_pathwise_loss(model, state, keys):
    losses = jax.vmap(lambda key: pathwise_loss(model, state, key))(keys)
    return losses.mean()


keys = jax.random.split(batch_key, 32)
loss_value, grads = eqx.filter_value_and_grad(batch_pathwise_loss)(model, state, keys)
```

## Checklist

- Put `simulate` inside the differentiated objective.
- Read a differentiable quantity from the final state or complete history.
- Keep the simulated state and objective live; do not call `stop_gradient` on them.
- Reuse one key within a value-and-gradient evaluation and split new keys between updates.
- Average multiple keys when a stochastic pathwise gradient is too noisy.
- Treat gradients through hard discrete structure as straight-through, biased surrogates.
- Do not add a score-function gradient for the same sampled discrete choice.
