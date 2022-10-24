import jax.numpy as np
from jax import lax, jit
from jax_md import partition

from jax_morph.datastructures import CellState

def simulation(fstep, params, fspace):
    
    n_ops = len(fstep)
    neighbor_list_fn = partition.neighbor_list(fspace.displacement, fspace.box_size, params["r_cutoff"])
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
        nbrs = neighbor_list_fn.allocate(new_istate.position)
        return new_istate, nbrs
    
    

    def sim_step(state, nbrs):
                        
        #first step must always be cell division
        state, logp = fstep[0](state, params, fspace, nbrs)

        for i in range(1,n_ops):
            state = fstep[i](state, params, fspace, nbrs)
        
        return state, logp
    
        nbrs = nbrs.update(state.position)
        if nbrs.did_buffer_overflow:  # Couldn't fit all the neighbors into the list.
            nbrs = neighbor_list_fn.allocate(state.position)

    return sim_init, sim_step

