import jax.numpy as np
from jax import jit, lax

import jax_md.dataclasses as jax_dataclasses

from jax_morph.datastructures import CellState
from jax_morph.cell_division import S_cell_division
from jax_morph.cell_growth import S_grow_cells

from .mechanical import S_mechmin_twotypes
from .secdiff import S_ss_chemfield
from .divrates import S_set_divrate



def _create_onecell_state(key, params):
    
    N = np.int16(params['ncells_init'])
    
    celltype = np.zeros(N, dtype=np.int16)
    celltype = celltype.at[0].set(1)
    
    radius = np.zeros(N, dtype=np.float32)
    radius = radius.at[0].set(params['cellRad'])
    
    position = np.zeros((N,2), dtype=np.float32)
    
    chemical = np.zeros((N,params['n_chem']), dtype=np.float32)
    field = np.zeros(N, dtype=np.float32)
    divrate = np.zeros(N, dtype=np.float32)
    divrate = divrate.at[0].set(1.)
        
    onec_state = CellState(position, celltype, radius, chemical, field, divrate, key)
    
    return onec_state



def init_state_grow(key, params, fspace):
    
    N = np.int16(params['ncells_init'])
    
    n_init_tot = params['ncells_init']
    n_init_ones = params['n_ones_init']
    
    onecell_state = _create_onecell_state(key, params)
    
    def _init_add(onecell_state, i):
        
        onecell_state, _ = S_cell_division(onecell_state, params, fspace)
        onecell_state = S_grow_cells(onecell_state, params, fspace)
        onecell_state = S_mechmin_twotypes(onecell_state, params, fspace)
    
        return onecell_state, 0.
    
    iterations = np.arange(params['ncells_init'])
    state, _ = lax.scan(_init_add, onecell_state, iterations)
    
    
    # divide cells in their subtypes arbitrarily
    celltype = np.zeros(N, dtype=np.int16)
    celltype = celltype.at[:n_init_ones].set(1)
    celltype = celltype.at[n_init_ones:n_init_tot].set(2)
    
    
    #set all cells to max radius and relax the system
    radius = np.zeros(N, dtype=np.float32)
    radius = radius.at[:n_init_tot].set(params['cellRad'])
    state = jax_dataclasses.replace(state, celltype=celltype, radius=radius)
    
    state = S_mechmin_twotypes(state, params, fspace)
    # calculate consistent chemfield
    state = S_ss_chemfield(state, params, fspace)
    #calculate consistent division rates
    state = S_set_divrate(state, params, fspace)

    
    return state
    