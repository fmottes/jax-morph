import jax.numpy as np
from jax import lax, jit

from jax_morph.datastructures import CellState


#TODO: define function signatures



def simulation(fstep, params, fspace):
    
    n_ops = len(fstep)
    #fstep = iter(fstep)
    
    def sim_init(istate, ncells_add=100, key=None):
        '''
        If key is none use the key packed in initial state, else use the provided key.
        '''

        ### elongate data structures to account for cells to be added

        #ncells_add = params['ncells_add']

        new_position = np.concatenate([istate.position, np.zeros((ncells_add,2))])
        new_chemical = np.concatenate([istate.chemical, np.zeros((ncells_add,params['n_chem']))])
        new_celltype = np.concatenate([istate.celltype, np.zeros(ncells_add)])
        new_radius = np.concatenate([istate.radius, np.zeros(ncells_add)])
        new_field = np.concatenate([istate.field, np.zeros(ncells_add)])
        new_divrate = np.concatenate([istate.divrate, np.zeros(ncells_add)])

        if None != key:
            new_key = key
        else:
            new_key = istate.key
            
        new_istate = CellState(new_position, new_celltype, new_radius, new_chemical, new_field, new_divrate, new_key)
        
        return new_istate
    
    

    def sim_step(state):
                        
        #first step must always be cell division
        state, logp = fstep[0](state, params, fspace)
        
        for i in range(1,n_ops):
            state = fstep[i](state, params, fspace)
        
        return state, logp
    
    
    return sim_init, sim_step




def sim_trajectory(istate, sim_init, sim_step, ncells_add=100, key=None, history=False):
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
            
    key : PRNGKey
            Key for the JAX RNG. If None (default) consumes and repalces the key stored in istate.
            
    history : bool
            Whether to return all of the intermediate states in the simulations in addition to the log probabilities of cell divisions.
            
            
    Returns
    ------------
    
    fstate : CellState
            End state of the simulation.
            
    aux : np.ndarray
            Auxiliary data returned by the simulation.
            If history=False returns an array of log probability of cell division performed during the simulation.
            If history=True each entry is a tuple (state_t, logp_t) for all the steps in the simulation.
    
    '''
        
    state = sim_init(istate, ncells_add, key)
    
    if history:
        def scan_fn(state, i):
            state, logp = sim_step(state)
            return state, (state, logp)
        
    else:
        def scan_fn(state, i):
            state, logp = sim_step(state)
            return state, logp
    
    
    iterations = len(state.celltype)-len(istate.celltype)
    iterations = np.arange(iterations)
    fstate, aux = lax.scan(scan_fn, state, iterations)
    
    return fstate, aux