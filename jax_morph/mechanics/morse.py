import jax.numpy as np
from jax import jit, lax, vmap

from jax_md import energy
import jax_md.dataclasses as jdc

from jax_morph.mechanics.minimizers import mechmin_sgd

from functools import partial



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
    #minimum of the well is (approx) at the sum of the two radii
    radii = np.array([state.radius-params['cellRad']*.02]) 
    sigma_matrix = radii+radii.T

    #calculate epsilon (well depth) for each pair based on type
    celltypeOne = np.array([np.where(state.celltype==1,1,0)]) 
    celltypeTwo = np.array([np.where(state.celltype==2,1,0)]) 
    
    epsilon_matrix = np.outer(celltypeOne , celltypeOne)* epsilon_OneOne + \
                   np.outer(celltypeTwo , celltypeTwo)* epsilon_TwoTwo + \
                   np.outer(celltypeOne , celltypeTwo)* epsilon_OneTwo + \
                   np.outer(celltypeTwo, celltypeOne)* epsilon_OneTwo 

    return epsilon_matrix, sigma_matrix



def _generate_morse_params_onetype(state, params):
    '''
    Morse interaction params for each particle couple. 

    Returns:
      sigma_matrix: Distance between particles where the energy has a minimum.
      epsilon_matrix: Depth of Morse well. 
    '''

    if 'epsilon' in params:
        epsilon_OneOne = params['epsilon']
    else:
        epsilon_OneOne = 3.


    #minimum energy when the two cells barely touch
    #minimum of the well is (approx) at the sum of the two radii
    radii = np.array([state.radius-params['cellRad']*.02]) 
    sigma_matrix = radii+radii.T

    #calculate epsilon (well depth) for each pair based on type
    celltypeOne = np.array([np.where(state.celltype==1,1,0)]) 


    epsilon_matrix = np.outer(celltypeOne , celltypeOne)*epsilon_OneOne

    return epsilon_matrix, sigma_matrix



def build_morse_energy(state, params, fspace, morse_eps_sigma='onetype'):
    '''
    Build the Morse energy function for the current state and parameters. 
    '''

    if morse_eps_sigma == 'onetype':
        morse_eps_sigma = _generate_morse_params_onetype
    elif morse_eps_sigma == 'twotypes':
        morse_eps_sigma = _generate_morse_params_twotypes
    #else:
    # it is assumed that morse_eps_sigma is a function that returns the epsilon and sigma matrices

    epsilon_matrix, sigma_matrix = morse_eps_sigma(state, params)

    energy_morse = energy.morse_pair(fspace.displacement,
                                     alpha=params['alpha'],
                                     epsilon=epsilon_matrix,
                                     sigma=sigma_matrix, 
                                     r_onset=params['r_onset'], 
                                     r_cutoff=params['r_cutoff'])

    return energy_morse



##############################################################
# STATE UPDATES
##############################################################


def S_mech_morse_relax(state, params, fspace, dt=.001, morse_eps_sigma='onetype', n_steps=None):
    '''
    Minimize mechanical energy with SGD. 
    Energy is given by the Morse potential with parameters calculated for the two-celltypes case.

    morse_eps_sigma:
      'onetype': use the same epsilon for all particles
      'twotypes': use different epsilons for each particle type
      function: a function that returns the epsilon and sigma matrices

    '''

    n_steps = n_steps if n_steps is not None else params['mech_relaxation_steps']

    
    if morse_eps_sigma == 'onetype':
        morse_eps_sigma = _generate_morse_params_onetype
    elif morse_eps_sigma == 'twotypes':
        morse_eps_sigma = _generate_morse_params_twotypes
    #else:
    # it is assumed that morse_eps_sigma is a function that returns the epsilon and sigma matrices

    
    epsilon_matrix, sigma_matrix = morse_eps_sigma(state, params)
    
    energy_morse = energy.morse_pair(fspace.displacement,
                                     alpha=params['alpha'],
                                     epsilon=epsilon_matrix,
                                     sigma=sigma_matrix, 
                                     r_onset=params['r_onset'], 
                                     r_cutoff=params['r_cutoff'])
    
    
    new_position = mechmin_sgd(energy_morse, 
                               state.position, 
                               fspace.shift, 
                               n_steps,
                               dt)
    
    new_state = jdc.replace(state, position=new_position)

    return new_state


#backwards compatibility
S_mechmin_twotypes = partial(S_mech_morse_relax, morse_eps_sigma='twotypes')