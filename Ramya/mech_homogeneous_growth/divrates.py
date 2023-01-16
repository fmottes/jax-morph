import jax.numpy as np
from jax import jit, lax, vmap, jacrev
import jax_md.dataclasses as jax_dataclasses
from jax_md import partition, util, smap, space, energy, quantity
from jax_morph.utils import logistic, polynomial
from Francesco.chem_twotypes.mechanical import _generate_morse_params_twotypes
maybe_downcast = util.maybe_downcast

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

# Functions to calculate divrates

def logistic_divrates(stresses, params):
    """ Calculates divrates using logistic functions on stress."""
    divrates = logistic(stresses,params["div_gamma"][0],params["div_k"][0])
    divrates = np.where(stresses > 0, logistic(stresses,params["div_gamma"][1],params["div_k"][1]), divrates)
    return divrates 



def div_mechanical(state, params, fspace, **kwargs) -> np.array:
    """ Calculates divrates only based on stress."""
    stresses = stress(state, params, fspace)   
    # calculate "rates"
    if "growth_fn" in kwargs:
        div = kwargs["growth_fn"](stresses, params)
    else: 
        div = logistic_divrates(stresses, params)
    # create array with new divrates
    divrate = np.where(state.celltype>0,div, 0.0)
    max_divrate = logistic(state.field, 0.1, 25.0)
    divrate = np.multiply(max_divrate, divrate)

    # cells cannot divide if they are too small
    # constants are arbitrary, change if you change cell radius
    divrate = divrate*logistic(state.radius+.06, 50, params['cellRad'])
    
    return divrate

def div_combined(state, params, fspace, **kwargs) -> np.array:
    """ Calculates divrates based on both stress and chemicals."""
    divrate = div_mechanical(state, params, fspace, **kwargs)
    # Get product of chemical contributions
    vmap_logistic = vmap(logistic, (1,0, 0),(1))
    divrate = np.multiply(divrate, np.prod(vmap_logistic(state.chemical,
    params["div_gamma"][2:],params["div_k"][2:]),axis=1,dtype=np.float32)) 
    divrate = divrate*logistic(state.radius+.06, 50, params['cellRad'])
    return divrate


def S_set_divrate(state, params, fspace, divrate_fn=div_mechanical,**kwargs):
    """ Sets divrates."""
    divrate = divrate_fn(state, params, fspace, **kwargs)
    new_state = jax_dataclasses.replace(state, divrate=divrate)
    
    return new_state