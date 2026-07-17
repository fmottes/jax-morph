"""Top-level drivers for model rollout and stochastic-trace scoring.

``simulate`` scans ``Model.__call__`` for ``n_steps`` macro-steps. The forward pass is a pure
sampler and is pathwise-differentiable end to end. ``trajectory_logp`` scores a recorded rollout
while carrying a live reconstructed state across macro-steps. ``transition_logp`` scores one
transition conditioned on an explicitly observed state. The library ships no trainer.
"""

from collections.abc import Iterable

import equinox as eqx
import jax
import jax.numpy as jnp

from .state import BaseState
from .step import Model

__all__ = ['simulate', 'trajectory_logp', 'transition_logp']


@eqx.filter_jit
def simulate(
    model: Model,
    state: BaseState,
    n_steps: int,
    dt: float | jax.Array = 1.0,
    key: jax.Array | None = None,
    *,
    history: bool = False,
    checkpoint: bool = False,
) -> BaseState:
    """Roll out ``model`` for ``n_steps`` macro-steps of size ``dt``; return the trajectory or final.

    The forward pass is a pure sampler, pathwise-differentiable end to end. With ``history=True`` the
    result is the stacked complete state history ``s_0, ..., s_n`` (leading ``n_steps + 1`` axis);
    otherwise it is the final state. ``key`` is required only when the model has a stochastic step -
    passing ``key=None`` then raises rather than silently freezing the seed. Trace fields on history
    frame zero belong to the input state, not a transition, so use ``transition_logp`` only when an
    intermediate state is intentionally an observed conditioning boundary.

    Args:
        model: The `Model` to advance.
        state: Initial `BaseState`.
        n_steps: Number of macro-steps to scan.
        dt: Macro-step size; physical end time is ``n_steps * dt``. Defaults to 1.0.
        key: PRNG key. Required for a model containing stochastic steps. Defaults to None.
        history: Whether to return the complete history ``s_0, ..., s_n`` instead of only the final
            state. Defaults to False.
        checkpoint: Whether to rematerialize macro-steps during backpropagation. Defaults to False.

    Returns:
        The final `BaseState`, or a stacked complete history with leading axis ``n_steps + 1`` when
        ``history`` is true.

    Raises:
        ValueError: If ``key`` is omitted for a model containing stochastic steps.

    Note: Scoring.
        The result carries no aggregate log-probability. Request a complete history and pass it to
        `trajectory_logp` to score a sampled rollout.

    Example: Sample a trajectory
        ```python
        trajectory = jxm.simulate(model, state, 20, dt=0.1, key=key, history=True)
        ```
    """
    if key is None:
        if any(step.is_stochastic for step in model.steps):
            raise ValueError(
                'simulate() received key=None but the model has stochastic steps; pass a PRNG key.'
            )
        key = jax.random.PRNGKey(0)  # placeholder: a deterministic model never consumes it
    keys = jax.random.split(key, n_steps)

    def macro(carry, k):
        return model(carry, dt=dt, key=k)

    step_fn = eqx.filter_checkpoint(macro) if checkpoint else macro

    def body(carry, k):
        new_state = step_fn(carry, k)
        return new_state, (new_state if history else None)

    final, traj = jax.lax.scan(body, state, keys)
    if not history:
        return final
    return jax.tree_util.tree_map(
        lambda initial, recorded: jnp.concatenate((initial[None], recorded), axis=0), state, traj
    )


@eqx.filter_jit
def trajectory_logp(
    model: Model,
    history: BaseState,
    dt: float | jax.Array = 1.0,
    *,
    score: str | Iterable[int] | None = None,
    checkpoint: bool = False,
) -> jax.Array:
    """Score a recorded trace trajectory with a live reconstructed state carry.

    ``history`` is the stacked complete state sequence ``s_0, ..., s_T`` returned by
    ``simulate(..., history=True)``. Frame zero is the detached initial conditioning state; frames
    one onward are detached recorded transition data. Deterministic computations and the replay
    semantics selected by each stochastic step remain live, including across numerical macro-step
    boundaries. ``score=None`` selects stochastic steps whose ``score_by_default`` is true,
    ``score='all'`` selects every stochastic step, and an explicit iterable selects model-step
    indices. ``checkpoint=True`` rematerializes each replayed macro-step during the backward pass.

    Args:
        model: Model whose stochastic steps produced the history.
        history: Complete stacked state history ``s_0, ..., s_T``.
        dt: Macro-step size used to produce the history. Defaults to 1.0.
        score: Stochastic-step selection: None, ``'all'``, or an iterable of model-step indices.
            Defaults to None (score the stochastic steps whose ``score_by_default`` is true).
        checkpoint: Whether to rematerialize replayed macro-steps during backpropagation. Defaults
            to False.

    Returns:
        Per-macro-step log-probability terms with shape ``(T,)``. This is not the scalar joint
        log-probability; call ``terms.sum()`` when a joint score is required.
    """
    scored = model._resolve_score(score)
    history = jax.lax.stop_gradient(history)
    initial_state = jax.tree_util.tree_map(lambda x: x[0], history)
    trajectory = jax.tree_util.tree_map(lambda x: x[1:], history)

    def replay_one(live_state, recorded_next):
        return model._replay_transition(live_state, recorded_next, dt, scored)

    step_fn = eqx.filter_checkpoint(replay_one) if checkpoint else replay_one
    _, terms = jax.lax.scan(step_fn, initial_state, trajectory)
    return terms


@eqx.filter_jit
def transition_logp(
    model: Model,
    observed_state: BaseState,
    observed_next: BaseState,
    dt: float | jax.Array = 1.0,
    *,
    score: str | Iterable[int] | None = None,
) -> jax.Array:
    """Score one recorded transition conditioned on an explicitly observed state.

    ``observed_state`` is detached as a statistical conditioning boundary. Stochastic traces are
    read from the detached ``observed_next`` state and replayed with parameters live. ``score`` has
    the same selection semantics as ``trajectory_logp``.

    Args:
        model: Model whose stochastic steps produced the transition.
        observed_state: State treated as the detached conditioning boundary.
        observed_next: Recorded next state containing the stochastic trace.
        dt: Macro-step size used to produce the transition. Defaults to 1.0.
        score: Stochastic-step selection: None, ``'all'``, or an iterable of model-step indices.
            Defaults to None (score the stochastic steps whose ``score_by_default`` is true).

    Returns:
        Scalar sum of the selected stochastic-step log-probabilities.
    """
    scored = model._resolve_score(score)
    observed_state = jax.lax.stop_gradient(observed_state)
    _, lp = model._replay_transition(observed_state, observed_next, dt, scored)
    return lp
