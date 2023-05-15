import jax
import jax.numpy as np
from jax import lax, jit

import jax_md.dataclasses as jdc


def simulation(fstep, params, fspace):
    
    n_ops = len(fstep)


    def sim_init(istate, ncells_add=0, key=None):
        '''
        If key is none use the key packed in initial state, else use the provided key.

        If ncells_add > 0 elongate the data structures to account for the cells to be added. 
        If ncells_add == 0 run one simulation step to initialize the data structures with consistent values and set random key (NO CELL DIVISION IS PERFORMED).

        '''

        if key is None:
            if istate.key is None:
                raise ValueError('No key provided for the RNG and no key found in the initial state.')
            else:
                key = istate.key


        ### elongate data structures to account for cells to be added
        if ncells_add > 0:

            new_fields = {}
            for field in jdc.fields(istate):

                if field.name == 'key':
                    new_fields['key'] = key
                else:
                    #retrieve the value of the field
                    value = getattr(istate, field.name)

                    if isinstance(value, np.ndarray):

                        if len(value.shape) > 0:
                            shape = (ncells_add,)+(value.shape[1:])
                            new_fields[field.name] = np.concatenate([value, np.zeros(shape, dtype=value.dtype)])
                            
                        else:
                            new_fields[field.name] = value
                    else:
                        new_fields[field.name] = value

            new_istate = type(istate)(**new_fields)

        elif 0 == ncells_add:
            new_istate = jdc.replace(istate, key=key)


        #run one "void" simulation step to initialize the data structures with consistent values
        #NO CELL DIVISION IS PERFORMED
        for i in range(1,n_ops):
            new_istate = fstep[i](new_istate, params, fspace)
        
        return new_istate
    
    

    def sim_step(state):
                        
        #first step must always be cell division
        state, logp = fstep[0](state, params, fspace)
        
        for i in range(1,n_ops):
            state = fstep[i](state, params, fspace)
        
        return state, logp
    
    
    return sim_init, sim_step




def sim_trajectory(istate, 
                   sim_init, 
                   sim_step, 
                   n_steps=100, 
                   key=None, 
                   history=False, 
                   init_multiplier=1., 
                   ncells_add=None):
    '''
    Runs a simulation trajectory for a given number of steps.
    The number of simulation steps is inferred from the size of the state datastructures before and after initialization.
    
    Args
    ------------
    
    istate : CellState
            Initial state of the system.
            
    sim_init : Callable
            Initialization function (created by the simulation.simulation function).
            
    sim_step : Callable
            Function performing one simulation step (created by the simulation.simulation function).

    n_steps : int
            Number of simulation steps to perform.

    init_mutiplier : float
            Multiplier for memory allocation during initialization.
            init_mutiplier=1. if only one cell division is expected at each step.
            
    key : PRNGKey
            Key for the JAX RNG. If None (default) consumes and repalces the key stored in istate.
            
    history : bool
            Whether to return all of the intermediate states in the simulations in addition to the log probabilities of cell divisions.

    n_cells_add : int
            Number of cells to add to the simulation.
            Same as n_steps, just there for backward compatibility.
            
            
    Returns
    ------------
    
    fstate : CellState
            End state of the simulation.
            
    aux : np.ndarray
            Auxiliary data returned by the simulation.
            If history=False returns an array of log probability of cell division performed during the simulation.
            If history=True each entry is a tuple (state_t, logp_t) for all the steps in the simulation.
    
    '''

    if ncells_add is not None:
        n_steps = ncells_add

        
    state = sim_init(istate, int(n_steps*init_multiplier), key)
    
    if history:
        def scan_fn(state, i):
            state, logp = sim_step(state)
            return state, (state, logp)
        
    else:
        def scan_fn(state, i):
            state, logp = sim_step(state)
            return state, logp
    
    
    iterations = np.arange(int(n_steps))
    fstate, aux = lax.scan(scan_fn, state, iterations)
    
    return fstate, aux