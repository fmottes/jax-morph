import jax
import jax.numpy as np
import equinox as eqx


def reinforce_loss(stop_rewards_grad=True):
    """
    Creates a loss function for the REINFORCE algorithm.

    Args:
        stop_rewards_grad (bool, optional): Whether to stop the gradient of the rewards. Defaults to True.

    Returns:
        function: A loss function that computes the REINFORCE loss given a model, SSA results, and rewards.



    The returned loss function has the following signature:

    Args:
        model: The model used to compute log probabilities.
        trajectory: The results from the SSA (Stochastic Simulation Algorithm).
        rewards: The rewards obtained from the environment.
        key_sim: The random key used to generate the trajectory.

    Returns:
        float: The computed REINFORCE loss.
    """

    def _loss(model, trajectory, rewards, key_sim):

        n_add = len(trajectory.position)
        subkeys_sim = np.asarray(jax.random.split(key_sim, n_add))

        trajectory = jax.tree_util.tree_map(lambda x: x[:-1], trajectory)

        call_model = lambda s, k: model(s, key=k)[1]
        log_ps = eqx.filter_vmap(call_model)(trajectory, subkeys_sim[1:])

        if stop_rewards_grad:
            rewards = jax.lax.stop_gradient(rewards)

        loss = -np.sum(log_ps * rewards)

        return loss

    return _loss
