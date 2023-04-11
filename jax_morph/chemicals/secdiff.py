import jax.numpy as np
from jax import jit, lax, vmap

import jax_md.dataclasses as jax_dataclasses

from jax_morph.chemicals.diffusion import diffuse_allchem


#new version of Findcss

#non jittable due to the bool mask based on celltype
#substitute with simulation step index to sidestep masking (not sure works either)

def S_ss_chemfield(state, params, fspace, sec_fn=None, diffusion_fn=diffuse_allchem, n_iter=5):
    '''
    Heuristically, steady state is reached in less than 5 iterations.
    '''
    
    if None == sec_fn:
        raise(ValueError('Need to pass a valid function for the calculation of the secretion rates.'))
    
    def _sec_diff_step(buff_state, i):
        
        #calculate new secretions
        sec = sec_fn(buff_state, params)
        
        #calculate new chemical concentrations
        chemfield = diffusion_fn(sec, buff_state, params, fspace)
        
        return jax_dataclasses.replace(buff_state, chemical=chemfield), 0.#, chemfield
    
    
    iterations = np.arange(n_iter)
    
    state, _ = lax.scan(_sec_diff_step, state, iterations)
    #uncomment line below and comment line above for history
    #new_state, chemfield = lax.scan(_sec_diff_step, new_state, iterations)

    return state