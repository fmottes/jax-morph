import jax.numpy as np
from jax import jit, lax, vmap

from jax_md import space, energy
import jax_md.dataclasses as jax_dataclasses



def diffuse_onechem(r,secRate,degRate,diffCoeff):
    '''
    NOTE: it is assumed that r is a pairwise distance
    '''
    diff = secRate/(2*np.sqrt(degRate*diffCoeff))*np.exp(-r*np.sqrt(degRate/diffCoeff))
    
    return diff


def diffuse_allchem(secretions, state, params, fspace):

    diff = energy.multiplicative_isotropic_cutoff(diffuse_onechem,
                                                  r_onset = params['r_onsetDiff'],
                                                  r_cutoff = params['r_cutoffDiff'])
    
    metric = space.metric(fspace.displacement)
    d = space.map_product(metric)
    
    #calculate all pairwise distances
    dist = d(state.position, state.position)
    
    # loop over all chemicals (vmap in future)
    new_chem = []
    for i in np.arange(params['n_chem']):
        
        c = diffuse_onechem(dist,
                            secretions[:,i],
                            params['degRate'][i],
                            params['diffCoeff'][i])
        
        c = c.sum(axis=1)
        
        #zero out concentration on empty sites
        c = np.where(state.celltype>0, c, 0.)
        
        c = np.reshape(c, (-1,1)) #make into column vector
        new_chem.append(c)
    
    new_chem = np.hstack(new_chem)
    
    return new_chem


def S_chem_diffusion(state, params, fspace=None, **kwargs):
    
    new_chemical = diffuse_allchem(state.chemical, state, params, fspace)
    
    new_state = jax_dataclasses.replace(state, chemical=new_chemical)
    
    return new_state