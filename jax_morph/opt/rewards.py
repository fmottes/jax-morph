import jax
import jax.numpy as np


def discounted_returns(rewards, GAMMA=0.97):
    """
    Calculate the discounted returns for a sequence of rewards.

    Args:
        rewards (array-like): A sequence of rewards.
        GAMMA (float, optional): The discount factor. Defaults to 0.97.

    Returns:
        numpy.ndarray: An array of discounted returns, where each element represents the cumulative discounted return from that point forward.
    """

    def discounting_add(carry, reward):
        return GAMMA * carry + reward, GAMMA * carry + reward

    # Reverse the rewards array
    reversed_rewards = np.flip(rewards, axis=0)

    # Initialize the carry to be the last element
    _, discounted_returns_reversed = jax.lax.scan(
        discounting_add, 0.0, reversed_rewards
    )
    discounted_returns = np.flip(discounted_returns_reversed, axis=0)

    return discounted_returns


def reward_ssq_diff(coordinate_idx=0):
    """Return a reward function that rewards increasing the sum of squares of the given coordinate. Rewards are calculated as the difference in the sum of squares between each time step.

    Args:
        coordinate_idx (int): The index of the coordinate to calculate the reward for.

    Returns:
        callable: The reward function.
    """

    def _rewards_fn(trajectory):
        """Calculate the ssq_diff rewards for the given trajectory.

        Args:
            trajectory (Trajectory): The SSA trajectory to calculate rewards for.

        Returns:
            ndarray: The rewards for each time step.
        """

        pos = trajectory.position[:, :, coordinate_idx]

        rewards = (pos**2).sum(axis=1)
        rewards = np.diff(rewards)

        return rewards

    return _rewards_fn
