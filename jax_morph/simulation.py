import jax
import jax.numpy as np
import jax_md

import equinox as eqx

from typing import Any, Callable, Union, Sequence
import abc



###------------BASE CELL STATE-----------------###

class BaseCellState(eqx.Module):
    '''
    Module containing the basic features of a system state.

    '''

    # METHODS
    displacement:   jax_md.space.DisplacementFn = eqx.field(static=True)
    shift:          jax_md.space.ShiftFn = eqx.field(static=True)

    # STATE
    position:   jax.Array
    celltype:   jax.Array
    radius:     jax.Array
    division:   jax.Array


    @classmethod
    def empty(cls, n_dim=2, n_cell_types=1):

        '''
        Intializes a CellState with no cells (empty data structures, with correct shapes).

        Parameters
        ----------
        n_dim: int
            Number of spatial dimensions.

        '''

        assert n_dim == 2 or n_dim == 3, 'n_dim must be 2 or 3'

        disp, shift = jax_md.space.free()
        

        args = {
            'displacement'  :   disp,
            'shift'         :   shift,
            'position'  :   np.empty(shape=(0, n_dim), dtype=np.float32),
            'celltype'  :   np.empty(shape=(0,n_cell_types), dtype=np.float32),
            'radius'    :   np.empty(shape=(0,1), dtype=np.float32),
            'division'   :   np.empty(shape=(0,1), dtype=np.float32),
            }
        
        return cls(**args)
    



###------------SIMULATION STEP ABC-----------------###


class SimulationStep(eqx.Module):

    @abc.abstractmethod
    def __call__(self, state, *, key=None, **kwargs):
        pass

    @abc.abstractmethod
    def return_logprob(self) -> bool:
        pass




###------------SEQUENTIAL SIMULATION STEP-----------------###


class Sequential(SimulationStep):
    substeps: tuple
    _return_logp: bool = eqx.field(static=True)


    def return_logprob(self) -> bool:
        return self._return_logp



    def __init__(self, substeps: Sequence[Callable]):

        if all(isinstance(x, SimulationStep) for x in substeps):
            self.substeps = tuple(substeps)
        else:
            raise TypeError("All substeps must be of type `SimulationStep`")
        
        self._return_logp = any(x.return_logprob() for x in self.substeps)
        


    @jax.named_scope("jax_morph.Sequential")
    def __call__(self, state, *, key=None, **kwargs):


        if key is None:
            keys = [None] * len(self.substeps)
        else:
            keys = jax.random.split(key, len(self.substeps))


        logp = np.float_(0.)

        for substep, key in zip(self.substeps, keys):

            if substep.return_logprob():
                state, logp = substep(state, key=key)
                logp += logp
            else:
                state = substep(state, key=key)


        if self._return_logp:
            return state, logp
        else:
            return state
        



    def __getitem__(self, i: Union[int, slice]) -> Callable:
        if isinstance(i, int):
            return self.substeps[i]
        elif isinstance(i, slice):
            return Sequential(self.substeps[i])
        else:
            raise TypeError(f"Indexing with type {type(i)} is not supported")

    def __iter__(self):
        yield from self.substeps

    def __len__(self):
        return len(self.substeps)