import jax.numpy as np
from jax import jit, vmap

from jax_md import space, energy
import jax_md.dataclasses as jdc



def diffuse_onechem_ss_exp(r,secRate,degRate,diffCoeff):
    '''
    NOTE: it is assumed that r is a pairwise distance
    '''
    diff = secRate/(2*np.sqrt(degRate*diffCoeff))*np.exp(-r*np.sqrt(degRate/diffCoeff))
    
    return diff


def diffuse_allchem_ss_exp(secretions, state, params, fspace):

    # diff = energy.multiplicative_isotropic_cutoff(diffuse_onechem_ss,
    #                                               r_onset = params['r_onsetDiff'],
    #                                               r_cutoff = params['r_cutoffDiff'])
    
    metric = space.metric(fspace.displacement)
    d = space.map_product(metric)
    
    #calculate all pairwise distances
    dist = d(state.position, state.position)
    
    # loop over all chemicals (vmap in future)
    new_chem = []
    for i in np.arange(secretions.shape[1]):
        
        c = diffuse_onechem_ss(dist,
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





def diffuse_allchem_ss(secretions, state, params, fspace):
    
    #calculate all pairwise distances
    dist = space.map_product(space.metric(fspace.displacement))(state.position, state.position)

    #prevent division by zero
    dist *= np.where(np.outer(state.celltype, state.celltype)>0, 1, -1)
    dist -= np.eye(dist.shape[0])

    #adjacency matrix
    # zero out connections to inexistent cells
    A = 1/dist
    A = (np.where(A>0, A, 0))**2


    #graph laplacian
    L = np.diag(np.sum(A, axis=0)) - A


    def ss_chemfield(P, D, K):

        #update laplacian
        L = D*L + K*np.eye(L.shape[0])

        #solve for steady state
        x = np.linalg.solve(L, P)

        return x

    new_chem = vmap(ss_chemfield, in_axes=(0, 0, 1), out_axes=(1,))(params['diffCoeff'], params['degRate'], secretions)

    return new_chem




def S_chem_diffusion(state, params, fspace=None, diffusion_fn=diffuse_allchem_ss):
    
    new_chemical = diffusion_fn(state.chemical, state, params, fspace)
    
    new_state = jdc.replace(state, chemical=new_chemical)
    
    return new_state