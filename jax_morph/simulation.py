import jax.numpy as np
from jax import lax, jit

from jax_morph.datastructures import CellState



def simulation(fstep, params, fspace):
    
    n_ops = len(fstep)
    #fstep = iter(fstep)
    
    def sim_init(istate, key=None):
        '''
        If key is none use the key packed in initial state, else use the provided key.
        '''

        ### elongate data structures to account for cells to be added

        ncells_add = params['ncells_add']

        new_position = np.concatenate([istate.position, np.zeros((ncells_add,2))])
        new_chemical = np.concatenate([istate.chemical, np.zeros((ncells_add,params['n_chem']))])
        new_celltype = np.concatenate([istate.celltype, np.zeros(ncells_add)])
        new_radius = np.concatenate([istate.radius, np.zeros(ncells_add)])
        new_divrate = np.concatenate([istate.divrate, np.zeros(ncells_add)])

        if None != key:
            new_key = key
        else:
            new_key = istate.key

        new_istate = CellState(new_position, new_celltype, new_radius, new_chemical, new_divrate, new_key)

        return new_istate
    
    

    def sim_step(state):
                        
        #first step must always be cell division
        state, logp = fstep[0](state, params, fspace)
        
        for i in range(1,n_ops):
            state = fstep[i](state, params, fspace)
        
        return state, logp
    
    
    return sim_init, sim_step




def sim_trajectory(istate, sim_init, sim_step, key=None):
    
    state = sim_init(istate, key)
    
    def scan_fn(state, i):
        state, logp = sim_step(state)
        return state, logp
    
    iterations = len(state.celltype)-len(istate.celltype)
    iterations = np.arange(iterations)
    state, logp = lax.scan(scan_fn, state, iterations)
    
    return state, logp