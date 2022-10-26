from IPython.core.formatters import ForwardDeclaredInstance
import jax.numpy as np
from jax import jit, lax, vmap, jacrev

import jax_md.dataclasses as jax_dataclasses
from jax_md import partition, util, smap, space, energy, quantity
from jax_morph.utils import logistic
maybe_downcast = util.maybe_downcast

def stress(fspace, state, sigma, epsilon, alpha, r_onset, r_cutoff):
    energy_fn = energy.morse_pair(fspace.displacement, epsilon=epsilon, alpha=alpha, sigma=sigma, r_onset=r_onset, r_cutoff=r_cutoff)
    force_fn = quantity.force(energy_fn)
    forces = force_fn(state.position)
    drs = space.map_product(fspace.displacement)(state.position, state.position)
    stresses = np.sum(np.multiply(forces, np.sign(drs)), axis=(0, -1))
    stresses = np.where(state.celltype > 0, stresses, 0.0)
    return stresses


def _generate_morse_params_twotypes(state, params):
    '''
    Morse interaction params for each particle couple. 

    Returns:
      sigma_matrix: Distance between particles where the energy has a minimum.
      epsilon_matrix: Depth of Morse well. 
    '''

    epsilon_OneOne = params['eps_OneOne']
    epsilon_TwoTwo = params['eps_TwoTwo']
    epsilon_OneTwo = params['eps_OneTwo']


    #minimum energy when the two cells barely touch
    #minimum of the well is (approx) at the sum of the two radii (sure??)
    radii = np.array([state.radius]) 
    sigma_matrix = radii+radii.T

    #calculate epsilon (well depth) for each pair based on type
    celltypeOne = np.array([np.where(state.celltype==1,1,0)]) 
    celltypeTwo = np.array([np.where(state.celltype==2,1,0)]) 
    
    epsilon_matrix = np.outer(celltypeOne , celltypeOne)* epsilon_OneOne + \
                   np.outer(celltypeTwo , celltypeTwo)* epsilon_TwoTwo + \
                   np.outer(celltypeOne , celltypeTwo)* epsilon_OneTwo + \
                   np.outer(celltypeTwo, celltypeOne)* epsilon_OneTwo 

    return epsilon_matrix, sigma_matrix

def div_mechanical(state, params, fspace, nbrs) -> np.array:
    div_gamma = params['div_gamma']
    div_k = params['div_k']
    
    # Calculate stresses
    epsilon_matrix, sigma_matrix = _generate_morse_params_twotypes(state, params)
    box_size = quantity.box_size_at_number_density(params['ncells_init'] + params['ncells_add'], 1.2, 2)
    stresses = stress(fspace, state, sigma_matrix, epsilon_matrix, params["alpha"], params["r_onset"], params["r_cutoff"])
    
    # calculate "rates"
    div = logistic(stresses,div_gamma[0],div_k[0])
    div = np.where(stresses > 0, logistic(stresses,div_gamma[1],div_k[1]), div)

    # create array with new divrates
    divrate = np.where(state.celltype>0,div, 0.0)
    max_divrate = logistic(state.chemical[:, 0], 0.1, 25.0)
    divrate = np.multiply(max_divrate, divrate)

    # cells cannot divide if they are too small
    # constants are arbitrary, change if you change cell radius
    divrate = divrate*logistic(state.radius+.06, 50, params['cellRad'])
    
    return divrate

def S_set_divrate(state, params, fspace, nbrs):
    
    divrate = div_mechanical(state, params, fspace, nbrs)
    new_state = jax_dataclasses.replace(state, divrate=divrate)
    
    return new_state