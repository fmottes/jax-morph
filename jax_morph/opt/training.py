import jax
import jax.numpy as np

import optax
import equinox as eqx

from tqdm import trange
from collections import namedtuple


def train(
    model,
    istate,
    loss,
    *,
    key,
    epochs=10,
    optimizer=optax.adam(1e-3),
    opt_model_filter=eqx.is_array,
    model_save_every=None,
    grad_save_every=None,
    normalize_grads=False,
    constrain_model=None,
    **kwargs,
):

    loss_and_grad = eqx.filter_jit(
        eqx.filter_value_and_grad(loss.loss_fn, has_aux=loss.has_aux)
    )

    key, subkey = jax.random.split(key)
    rl, g = loss_and_grad(model, istate, key=subkey)

    if loss.has_aux:
        rl, l = rl
        losses = [float(l)]
    else:
        losses = None

    rlosses = [float(rl)]

    if model_save_every is not None:
        models = [model]
    else:
        models = None

    if grad_save_every is not None:
        grads = [g]
    else:
        grads = None

    opt_state = optimizer.init(eqx.filter(model, opt_model_filter))

    pbar = trange(epochs)
    for e in pbar:
        try:

            ### MODEL UPDATE AND GRADIENT
            if normalize_grads:
                g = jax.tree_map(lambda x: x / (np.linalg.norm(x) + 1e-10), g)

            updates, opt_state = optimizer.update(g, opt_state, model)
            model = eqx.apply_updates(model, updates)
            if constrain_model is not None:
                model = constrain_model(model)

            key, subkey = jax.random.split(key)
            rl, g = loss_and_grad(model, istate, key=subkey)

            ### LOGGING
            if loss.has_aux:
                rl, l = rl
                losses += [float(l)]

            rlosses += [float(rl)]

            if model_save_every is not None and e % model_save_every == 0:
                models += [model]

            if grad_save_every is not None and e % grad_save_every == 0:
                grads += [g]

            ### PROGRESS BAR
            if loss.has_aux:
                pbar.set_description(f"Loss: {l:.5f}")
            else:
                pbar.set_description(f"Epoch: {e}")

        except FloatingPointError:
            print("NaN or Overflow")
            break

        except KeyboardInterrupt:
            print("Interrupted")
            break

    return model, namedtuple(
        "OptimizationResults", ["model", "loss", "loss_aux", "grad"]
    )(models, rlosses, losses, grads)
