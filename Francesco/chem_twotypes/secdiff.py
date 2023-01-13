import jax.numpy as np
from jax import jit, lax, vmap

import jax_md.dataclasses as jax_dataclasses

from jax_morph.datastructures import CellState
from jax_morph.utils import logistic
from jax_morph.diffusion import diffuse_allchem



# Function for the secretion rate of one chemical 
#as a function of concentration of all other chemicals
def _sec_onechem(chem, mumax, gammavec, kvec):  
    """
    Helper function.
    Calculates secretion rate of one chemical by each cell 
    from the concentrations of the other chemicals.
    
    Arg:
    c : concetration of chemicals , an nCells x n_chem matrix
    mumax, gammavec, kvec : parameters for the logistic functions
    
    Returns:
    sec_onechem : 
        nCells x 1 array of secretion rates of one chemical by each cell
    """

    vmap_logistic = vmap(logistic, (1,0,0),(1))

    sec_onechem = mumax*np.prod(vmap_logistic(chem,gammavec,kvec), 
                                         axis=1, 
                                         dtype=np.float32)

    return sec_onechem



#(sec x conc) x (conc x cell)
# Function that returns secretion rates given current concentration
def sec_chem_logistic(state, params):
    """
    Calculates secretion rate of chemicals by each cell.
    
    Args:
    state: current state
    params: dictionary with parameters
    
    Returns:
    secRates: secretion rates
    """


    sec_max = params['sec_max']
    sec_gamma = params['sec_gamma']
    sec_k = params['sec_k']
    sec_by_ctypes = params['secreted_by_ctypes']

    
    #generalize secretion to n_chem cell types
    #each cell type secretes only one chemical
    
    sec_all = []
    
    for c in np.arange(params['n_chem'], dtype=np.int16):
                
        sec_onec = _sec_onechem(state.chemical, sec_max[c], sec_gamma[c,:], sec_k[c,:])
        
        #set sec to zero everywhere but where the secreting ctypes are
        cts = np.array(sec_by_ctypes[c], dtype=int) #cast for safety
        sec_onec = np.where(np.isin(state.celltype, cts), sec_onec, 0.)
        
        #make into column vector
        sec_onec = np.reshape(sec_onec, (-1,1))
        
        sec_all.append(sec_onec)
        
    #(cells x chemical) matrix with secretion
    sec_all = np.concatenate(sec_all, axis=1)
    

    return sec_all



#new version of Findcss

#non jittable due to the bool mask based on celltype
#substitute with simulation step index to sidestep masking (not sure works either)

def S_ss_chemfield(state, params, fspace, sec_chemical=sec_chem_logistic, n_iter=5):
    '''
    Heuristically, steady state is reached in less than 5 iterations.
    '''
    def _sec_diff_step(buff_state, i):
        
        #calculate new secretions
        sec = sec_chemical(buff_state, params)
        
        #calculate new chemical concentrations
        chemfield = diffuse_allchem(sec, buff_state, params, fspace)
        
        return jax_dataclasses.replace(buff_state, chemical=chemfield), 0.#, chemfield
    
    #buffer state for looping (not strictly necessary)
    new_state = CellState(*jax_dataclasses.unpack(state))
    
    iterations = np.arange(n_iter)
    
    new_state, _ = lax.scan(_sec_diff_step, new_state, iterations)
    #uncomment line below and comment line above for history
    #new_state, chemfield = lax.scan(_sec_diff_step, new_state, iterations)

    return new_state