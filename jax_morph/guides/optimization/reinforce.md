# REINFORCE training

Use a score-function estimator when the reward depends on a non-differentiable sampled event, such
as a terminal cell count. A REINFORCE update has two distinct stages:

1. **Sample outside the differentiated loss.** Roll out the current model with
   `simulate(..., history=True)` and detach the resulting history.
2. **Score inside the differentiated loss.** Pass that fixed history to `trajectory_logp` with the
   candidate model live, and differentiate the scalar score surrogate.

This separation is the recommended pattern. `simulate` belongs inside a differentiated objective
for [pathwise training](pathwise.md), but not inside the REINFORCE score surrogate.

jax-morph supplies the sampler and trajectory scorer. It deliberately leaves rewards, batching,
baselines, optimizers, and training loops to user code. Install an optimizer separately if needed:

```bash
uv add optax
```

## One rollout

First sample a complete history from the current model:

```python
import equinox as eqx
import jax
import jax.numpy as jnp
import jax_morph as jxm


N_STEPS = 100
DT = 0.1


def sample_rollout(model, state, key):
    history = jxm.simulate(
        model,
        state,
        n_steps=N_STEPS,
        dt=DT,
        key=key,
        history=True,
    )
    return jax.lax.stop_gradient(history)


history = sample_rollout(model, state, key)
reward = -jnp.square(history.alive[-1].sum() - 32)
advantage = reward - baseline
```

Then differentiate only the score surrogate. Equinox differentiates the first argument, so the
recorded history and advantage are data while `candidate_model` remains live:

```python
def reinforce_loss(candidate_model, recorded_history, advantage):
    logp_terms = jxm.trajectory_logp(candidate_model, recorded_history, dt=DT)
    return -jax.lax.stop_gradient(advantage) * logp_terms.sum()


loss_value, grads = eqx.filter_value_and_grad(reinforce_loss)(
    model,
    history,
    advantage,
)
```

The model used to sample `history` and the model differentiated by `reinforce_loss` must be the same
current model for an ordinary on-policy update. Compute the gradient before applying an optimizer
update. Scoring an older rollout under a changed model is off-policy and needs an explicit correction
such as importance weighting; it should not happen accidentally because a rollout was reused.

`trajectory_logp` detaches its history internally, but the explicit `stop_gradient` at the sampling
boundary is still useful: it documents that histories are data and keeps reward code from creating an
unintended pathwise gradient.

## Batched updates

Sample several independent histories before evaluating one score loss. A batch supports a useful
baseline and usually has much lower variance than a single rollout:

```python
def sample_batch(model, state, keys):
    histories = jax.vmap(lambda key: sample_rollout(model, state, key))(keys)
    return jax.lax.stop_gradient(histories)


def batch_reinforce_loss(candidate_model, recorded_histories, advantages):
    def score(history):
        return jxm.trajectory_logp(candidate_model, history, dt=DT).sum()

    joint_logp = jax.vmap(score)(recorded_histories)
    advantages = jax.lax.stop_gradient(advantages)
    return -jnp.mean(advantages * joint_logp)


keys = jax.random.split(batch_key, 32)
histories = sample_batch(model, state, keys)
counts = histories.alive[:, -1, :].sum(axis=-1)
rewards = -jnp.square(counts - 32)
advantages = rewards - running_baseline  # baseline estimated from previous batches
loss_value, grads = eqx.filter_value_and_grad(batch_reinforce_loss)(
    model,
    histories,
    advantages,
)
```

A constant or running-mean baseline from previous batches changes variance, not the expected
gradient. Do not use a rollout's own reward as its baseline. Detach learned baselines too unless
intentionally training them with a separate baseline loss.

## Terminal and shaped returns

`trajectory_logp` returns one term per macro-step, with shape `(n_steps,)` for one history. For a
terminal reward, multiply the reward or advantage by `logp_terms.sum()` as above.

For shaped rewards, construct a return-to-go for each transition and align its shape with the score
terms:

```python
# returns_to_go[t] contains rewards from transition t onward
loss = -jnp.sum(
    jax.lax.stop_gradient(returns_to_go - baselines) * logp_terms
)
```

Do not return `-advantage * logp_terms` without reducing it: that is a vector, not a scalar loss.
Frame zero of a complete history is the initial state, so `history` has length `n_steps + 1` while
`logp_terms` and transition-aligned returns have length `n_steps`.

## Why not put `simulate` inside this loss?

The following compact form can produce the same gradient because `trajectory_logp` detaches the
history and the reward is explicitly detached:

```python
def compact_but_discouraged(model, state, key):
    history = jxm.simulate(model, state, N_STEPS, DT, key, history=True)
    reward = terminal_reward(history)
    terms = jxm.trajectory_logp(model, history, DT)
    return -jax.lax.stop_gradient(reward) * terms.sum()
```

It is discouraged as a training pattern because it makes sampling occur whenever the loss is
evaluated, obscures which values are fixed observations, and makes it easy to accidentally retain a
pathwise gradient through the reward. Separating the two stages also lets one batch be scored,
normalized, inspected, and then discarded deliberately.

Both stages still have necessary work: `simulate` samples and records exact stochastic traces, then
`trajectory_logp` replays those traces with distribution parameters live. Replay is how gradients
can reach not only a stochastic step's direct parameters but also earlier deterministic or unscored
reparameterized computations that shaped its action probabilities across macro-steps.

## Checklist

- Sample with the current model and `history=True`.
- Stop the history gradient at the sampler/scorer boundary.
- Compute rewards and baselines from the recorded history, then detach the resulting advantages.
- Call `trajectory_logp` once per complete rollout; do not score adjacent transitions independently.
- Reduce the per-step score terms to one scalar surrogate.
- Differentiate the scorer with respect to the current model, then update the model.
- Do not add a straight-through pathwise gradient for the same sampled discrete choice.

The complete acceptance model in `tests/test_acceptance.py` shows a built physics-and-control
pipeline suitable for adapting to a training script, and the control example notebook contains a
batched Optax REINFORCE loop.
