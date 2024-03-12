import jax.numpy as np
from jax import vmap

import jax_md.dataclasses as jax_dataclasses


### STATE UPDATES ###
def S_chemical_gradients(state, params, fspace, R=None):
    '''
    Calculate chemical gradients based on current chemical concentrations.
    Gradients are given as an ncells x (grad_x + grad_y) matrix.
    '''
    
    chemgrad = chem_grad(state, fspace, R)
    
    state = jax_dataclasses.replace(state, chemgrad=chemgrad)

    return state




# ACTUAL GRADIENT CALCULATION
def chem_grad(state, fspace, R=None):
    '''
    JIT-able function that calculates cells' internal estimates of chemical gradients.

    Calculates grads of every chem conc. for every (alive) cell. Cells lying at a distance < R from one another are considered neigbors. If R=None calculates "touching" distance based on current cell radii.
    '''
    
    # mask only cells that exist
    c_alive = state.celltype>0.

    # displacements between cell pairs (ncells x ncells x ndim_space)
    disp = vmap(vmap(fspace.displacement, in_axes=[0,None]), in_axes=[None,0])(state.position, state.position)

    # distances btw cell pairs
    # dist w/ non-existing cells are zeroed out
    dist = np.sqrt((disp**2).sum(2))*np.outer(c_alive, c_alive)

    # consider as neigbors:
    # - cells less than one radius away (+ small tolerance)
    # - cells differents from themselves
    # - cells that exist
    if None == R:
        # "touching" distance betw. cells
        R = (state.radius+state.radius[:,None])*np.outer(c_alive, c_alive)
    
    neig = (dist<R)*(dist>.0)

    # normalize all displacements
    norm_disp = (disp*neig[:,:,None])/(dist[:,:,None]+1e-8)

    # calculates x and y components of grad for single chemical
    # GOD KNOWS HOW I CAME UP WITH THIS, BUT IT WORKS
    def _grad_chem(chem):
        # bincount supports only sums of scalar weights
        # vmap over components of chem gradients
        #return vmap(np.bincount, in_axes=[None,1], out_axes=1)(neig[0], chem[neig[1],None]*norm_disp)
        
        return (norm_disp*chem.ravel()[:,None]).sum(1)
        

    #vectorize over chemicals
    #OUTPUT SHAPE: ncells x ndim x nchem
    _grad_chem = vmap(_grad_chem, in_axes=1, out_axes=2)

    #calc grads (no non-existing cells or lone cells w/ no neighbors)
    grads = _grad_chem(state.chemical)
        
    # transform into ncells x (grad_x + grad_y)
    #reshape like ncells x ndim x nchem to revert
    grads = grads.reshape(len(state.celltype), -1)
    
    return grads










##############################################################
# NOT JIT-ABLE VERSION
##############################################################
#not jit-able due to the bool mask based on celltype
#not used in the code, but kept for reference

def chem_grad_nojit(state, fspace, R=None):
    '''
    NOT JIT-able function that calculates cells' internal estimates of chemical gradients.

    Calculates grads of every chem conc. for every (alive) cell. Cells lying at a distance < R from one another are considered neigbors. If R=None calculates "touching" distance based on current cell radii.
    '''
    
    # mask only cells that exist
    c_alive = state.celltype>0.

    # displacements between cell pairs (ncells x ncells x ndim_space)
    disp = vmap(vmap(fspace.displacement, in_axes=[0,None]), in_axes=[None,0])(state.position, state.position)

    # distances btw cell pairs
    # dist w/ non-existing cells are zeroed out
    dist = np.sqrt((disp**2).sum(2))*np.outer(c_alive, c_alive)

    # consider as neigbors:
    # - cells less than one radius away (+ small tolerance)
    # - cells differents from themselves
    # - cells that exist
    if None == R:
        # "touching" distance betw. cells
        R = (state.radius+state.radius[:,None])*np.outer(c_alive, c_alive)
    
    neig = np.nonzero((dist<R)*(dist>.0))

    # normalize all displacements
    norm_disp = disp[neig]/dist[neig][:,None]

    # calculates x and y components of grad for single chemical
    # GOD KNOWS HOW I CAME UP WITH THIS, BUT IT WORKS
    def _grad_chem(chem):
        # bincount supports only sums of scalar weights
        # vmap over components of chem gradients
        return vmap(np.bincount, in_axes=[None,1], out_axes=1)(neig[0], chem[neig[1],None]*norm_disp)

    #vectorize over chemicals
    #OUTPUT SHAPE: ncells x ndim x nchem
    _grad_chem = vmap(_grad_chem, in_axes=1, out_axes=2)

    #calc grads (no non-existing cells or lone cells w/ no neighbors)
    grads_alive = _grad_chem(state.chemical)

    # create data structure w/ zeros
    grads = np.zeros((state.celltype.shape[0],state.position.shape[1],state.chemical.shape[1]))
    
    # set calculated gradients where appropriate
    idx = np.unique(neig[0]) #needed in case bincount spits out some zeros too
    grads = grads.at[idx].set(grads_alive[idx])
    
    # transform into ncells x (grad_x + grad_y)
    #reshape like ncells x ndim x nchem to revert
    grads = grads.reshape(len(state.celltype), -1)
    
    return grads