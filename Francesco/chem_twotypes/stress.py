import jax.numpy as np

from jax import jacrev
from jax_md import energy, space
import jax_md.dataclasses as jax_dataclasses

from .mechanical import _generate_morse_params_twotypes


def stress(state, params, fspace):
    """ Calculates stress on each cell."""
    
    epsilon_matrix, sigma_matrix = _generate_morse_params_twotypes(state, params)
    
    energy_fn = energy.morse_pair(fspace.displacement, epsilon=epsilon_matrix, alpha=params["alpha"], sigma=sigma_matrix, r_onset=params["r_onset"], r_cutoff=params["r_cutoff"], per_particle=True)
    
    # Removed the minus sign because we want F_ij = force on i (not by i)
    forces = jacrev(energy_fn)(state.position)
    
    # F_ij = force on i by j, r_ij = displacement from i to j
    drs = space.map_product(fspace.displacement)(state.position, state.position)
    
    stresses = np.sum(np.multiply(forces, np.sign(drs)), axis=(0, 2))
    stresses = np.where(state.celltype > 0, stresses, 0.0)
    
    return stresses



def S_set_stress(state, params, fspace, stress_fn=stress):
    
    stresses = stress_fn(state, params, fspace)
    
    new_state = jax_dataclasses.replace(state, stress=stresses)
    
    return new_state