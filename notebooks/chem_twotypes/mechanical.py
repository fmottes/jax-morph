import jax.numpy as np
from jax import jit, lax, vmap

from jax_md import energy
import jax_md.dataclasses as jax_dataclasses

from jax_morph.mechanics.minimizers import mechmin_sgd




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




def S_mechmin_twotypes(state, params, fspace, dt=.001):
    '''
    Minimize mechanical energy with SGD. 
    Energy is given by the Morse potential with parameters calculated for the two-celltypes case.
    '''
    
    
    epsilon_matrix, sigma_matrix = _generate_morse_params_twotypes(state, params)
    
    energy_morse = energy.morse_pair(fspace.displacement,
                                     alpha=params['alpha'],
                                     epsilon=epsilon_matrix,
                                     sigma=sigma_matrix, 
                                     r_onset=params['r_onset'], 
                                     r_cutoff=params['r_cutoff'])
    
    
    new_position = mechmin_sgd(energy_morse, state, params, fspace, dt)
    
    new_state = jax_dataclasses.replace(state, position=new_position)

    return new_state