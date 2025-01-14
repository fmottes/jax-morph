import jax.numpy as np


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
