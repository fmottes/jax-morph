import jax
import jax.numpy as np
import equinox as eqx
import optax
from tqdm import trange

from ..simulation import simulate
from .rewards import discounted_returns

from typing import NamedTuple, Tuple


class ReinforceTrainingLog(NamedTuple):
    """A named tuple containing training statistics and hyperparameters from REINFORCE training.
    Attributes:
        losses (list): History of loss values during training
        rewards (list): History of reward values during training
        saved_models (dict): Dictionary of trained models at different epochs
        istate (eqx.Module): Initial state configuration
        epochs (int): Number of training epochs
        learning_rate (float): Learning rate used for training
        batch_size (int): Batch size used during training
        return_discount (float): Discount factor for return calculation
        optimizer (str): Name of optimizer used for training
    """

    losses: list
    rewards: list
    static_model: eqx.Module
    saved_models: dict
    istate: eqx.Module
    epochs: int
    learning_rate: float
    batch_size: int
    return_discount: float
    optimizer: str
    keyboard_interrupt: bool

    def model_at_epoch(self, epoch: int) -> eqx.Module:
        """Returns the model saved at a given epoch."""

        if epoch not in self.saved_models:
            raise ValueError(f"Epoch {epoch} not found in saved models")

        params_model = self.saved_models[epoch]
        return eqx.combine(params_model, self.static_model)


def train_reinforce(
    key,
    model,
    istate,
    rewards_fn,
    loss_fn,
    epochs=20,
    learning_rate=1e-3,
    batch_size=4,
    return_discount=0.97,
    optimizer=optax.adam,
    save_model_every=None,
) -> Tuple[eqx.Module, ReinforceTrainingLog]:
    """Trains a model using policy gradients class of algorithms.

    Safely interrupts training when hit with KeyboardInterrupt.

    Args:
        key (jax.random.PRNGKey): Random number generator key for reproducibility.
        model (eqx.Module): The model to be trained.
        istate (InitialState): Initial state object containing starting conditions.
        rewards_fn (Callable): Function that computes rewards given simulation results.
        loss_fn (Callable): Function that computes loss given model, simulation results, returns and random keys.
        epochs (int, optional): Number of training epochs. Defaults to 20.
        learning_rate (float, optional): Learning rate for optimization. Defaults to 1e-3.
        batch_size (int, optional): Number of trajectories per batch. Defaults to 4.
        return_discount (float, optional): Discount factor for computing returns. Defaults to 0.97.
        optimizer (Callable, optional): Optax optimizer to use. Defaults to optax.adam.
        save_model_every (int, optional): Save model every N epochs. If None, only saves final model. Defaults to None.

    Returns:
        tuple: A tuple containing:
            - model (eqx.Module): The trained model
            - log (ReinforceTrainingLog): Training log containing losses, rewards, saved models and training parameters
    """

    N_ADD = int(istate.celltype.shape[0] - istate.celltype.sum(-1).sum(-1))

    # JIT-compiled functions
    simulate_batch = eqx.filter_jit(
        eqx.filter_vmap(
            lambda model, batch_keys: simulate(
                model, istate, batch_keys, n_steps=N_ADD, history=True
            )[0],
            in_axes=(None, 0),
        )
    )

    batch_rewards = eqx.filter_jit(eqx.filter_vmap(rewards_fn, in_axes=0))
    loss_and_grads = eqx.filter_jit(
        eqx.filter_vmap(eqx.filter_value_and_grad(loss_fn), in_axes=(None, 0, 0, 0))
    )

    # JIT the update step
    @eqx.filter_jit
    def update_step(model, opt_state, batch_subkeys):

        # Run simulation
        sim_results = simulate_batch(model, batch_subkeys)

        # Calculate rewards
        rewards = batch_rewards(sim_results)

        # Calculate returns
        returns = eqx.filter_vmap(discounted_returns, in_axes=(0, None))(
            rewards, return_discount
        )

        # Calculate loss and gradients per trajectory
        loss, grads = loss_and_grads(model, sim_results, returns, batch_subkeys)
        grads = jax.tree.map(lambda x: x.mean(axis=0), grads)  # Average over batch

        updates, new_opt_state = opt.update(grads, opt_state, model)

        new_model = eqx.apply_updates(model, updates)

        return new_model, new_opt_state, loss.mean(), rewards

    losses = []
    reward = []
    saved_models = {}

    opt = optimizer(learning_rate)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    _, static_model = eqx.partition(model, eqx.is_array)

    keyboard_interrupt = False
    pbar = trange(epochs)
    for epoch in pbar:

        try:
            key, *subkeys = jax.random.split(key, batch_size + 1)
            batch_subkeys = np.array(subkeys)

            # Update model
            model, opt_state, loss, rewards = update_step(
                model, opt_state, batch_subkeys
            )

            if save_model_every is not None and epoch % save_model_every == 0:
                params_model, _ = eqx.partition(model, eqx.is_array)
                saved_models[epoch] = params_model

            losses += [float(loss)]
            reward += [float(rewards.sum(-1).mean())]
            pbar.set_description(f"Loss: {loss:.2f}, Reward: {reward[-1]:.2f}")

        except KeyboardInterrupt:
            keyboard_interrupt = True
            break

    saved_models[epoch] = eqx.partition(model, eqx.is_array)[0]

    if keyboard_interrupt:
        print(f"Training Interrupted after {epoch} epochs ({epoch/epochs*100:.2f}%)")
        epochs = epoch

    log = ReinforceTrainingLog(
        losses=losses,
        rewards=reward,
        static_model=static_model,
        saved_models=saved_models,
        istate=istate,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        return_discount=return_discount,
        optimizer=optimizer.__name__,
        keyboard_interrupt=keyboard_interrupt,
    )

    return model, log
