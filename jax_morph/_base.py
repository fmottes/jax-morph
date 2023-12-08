import jax
import jax.numpy as np
import jax_md

import equinox as eqx

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
    



###------------SIMULATION STEP ABC-----------------###


class SimulationStep(eqx.Module):

    @abc.abstractmethod
    def __call__(self, state, *, key=None, **kwargs):
        pass

    @abc.abstractmethod
    def return_logprob(self) -> bool:
        pass

