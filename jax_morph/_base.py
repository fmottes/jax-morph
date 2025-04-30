import jax
import jax.numpy as np
import jax_md.space

import equinox as eqx

import abc

from functools import partial
from typing import Optional, Union

# ------------BASE CELL STATE-----------------


class BaseCellState(eqx.Module):
    """
    Module containing the basic features of a system state.

    """

    # METHODS
    displacement: jax_md.space.DisplacementFn = eqx.field(static=True)
    shift: jax_md.space.ShiftFn = eqx.field(static=True)

    # STATE
    position: jax.Array
    celltype: jax.Array
    radius: jax.Array

    @classmethod
    def empty(cls, n_dim=2, n_cell_types=1):
        """
        Intializes a CellState with no cells (empty data structures, with correct shapes).

        Parameters
        ----------
        n_dim: int
            Number of spatial dimensions.

        """

        assert n_dim == 2 or n_dim == 3, "n_dim must be 2 or 3"

        disp, shift = jax_md.space.free()

        args = {
            "displacement": disp,
            "shift": shift,
            "position": np.empty(shape=(0, n_dim), dtype=np.float32),
            "celltype": np.empty(shape=(0, n_cell_types), dtype=np.float32),
            "radius": np.empty(shape=(0, 1), dtype=np.float32),
        }

        return cls(**args)

    def elongate(self, n_add):
        """Elongate state array fields to accomodate additional particles."""

        def _elongate(x):

            if isinstance(x, jax.Array):
                if x.ndim == 1:
                    x = np.concatenate([x, np.zeros(n_add)])
                else:
                    x = np.concatenate([x, np.zeros((n_add, *x.shape[1:]))], axis=0)

            return x

        return jax.tree_map(_elongate, self)

    def delete(self, del_idx):
        """Delete state array fields to do regeneration experiments."""

        def _delete(x, del_idx):

            if isinstance(x, jax.Array):
                x = np.delete(x, del_idx, axis=0)
            return x

        return jax.tree_map(partial(_delete, del_idx=del_idx), self)


# ------------SIMULATION STEP ABC-----------------


class SimulationStep(eqx.Module):
    """Abstract base class for simulation steps in a JAX-based morphological simulation.

    This class defines the interface for individual simulation steps that can modify
    a cell state. Each step can read from and write to specific fields of the state,
    tracked via _read_state_fields and _write_state_fields.

    Attributes:
        _read_state_fields: Fields this step needs to read from the state
        _write_state_fields: Fields this step may modify in the state
    """

    _read_state_fields: tuple[str, ...] = eqx.field(static=True)
    _write_state_fields: tuple[str, ...] = eqx.field(static=True)

    @abc.abstractmethod
    def return_logprob(self) -> bool:
        """
        Whether this step returns a log probability.

        Returns:
            bool: True if the step returns a log probability, False otherwise
        """
        pass

    @property
    def required_state_fields(self) -> tuple[str, ...]:
        """Get all state fields required by this simulation step.

        Returns:
            tuple[str, ...]: Combined set of read and write fields
        """
        return tuple(set(self._read_state_fields + self._write_state_fields))

    @property
    def read_state_fields(self):
        return self._read_state_fields

    @property
    def write_state_fields(self):
        return self._write_state_fields

    def check_state_fields(self, state: BaseCellState) -> None:
        """Validate that the state contains all required fields.

        Args:
            state: The cell state to validate

        Raises:
            ValueError: If any required field is missing from the state
        """
        for field in self.required_state_fields:
            if field not in state.__dict__:
                raise ValueError(f"Required state field {field} not found in state")

    @abc.abstractmethod
    def __call__(
        self,
        state: BaseCellState,
        *,
        key: Optional[jax.random.PRNGKey] = None,
        **kwargs,
    ) -> Union[BaseCellState, tuple[BaseCellState, jax.Array]]:
        """Abstract method for the simulation step.

        Args:
            state: The current state of the cell
            key: A JAX random key for stochastic operations
            **kwargs: Additional keyword arguments

        Returns:
            Union[BaseCellState, tuple[BaseCellState, jax.Array]]: Either:
                - The updated state of the cell
                - A tuple of (updated state, log probability) if return_logprob() is True
        """
        pass
