import jax.numpy as np
from jax import jit, lax, vmap, nn, jacrev
from jax_md import partition, util, smap, space, energy, quantity, dataclasses
from jax_morph.utils import logistic
from Francesco.chem_twotypes.mechanical import _generate_morse_params_twotypes
import haiku as hk
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

### Functions to calculate divrates

def logistic_divrates(stresses, params):
    """ Calculates divrates using logistic functions on stress."""
    divrates = logistic(stresses,params["div_gamma"][0],params["div_k"][0])
    divrates = np.where(stresses > 0, logistic(stresses,params["div_gamma"][1],params["div_k"][1]), divrates)
    return divrates 

def nn_divrates():
    """ Creates haiku NN that can be used to calculate divrates."""
    def nn_fun(cell_inputs):
        # cell inputs is an array of size [num inputs, num cells]
        mlp = hk.Sequential([
            hk.Linear(3), nn.relu,
            hk.Linear(1), nn.sigmoid,
        ])
        output = mlp(cell_inputs)
        return output
    nn_fun_t = hk.transform(nn_fun)
    # Returned object contains init,apply functions
    return nn_fun_t

###

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

def div_nn(state, params, fspace, nn_fun_t):
    # Cell inputs: stress, chemicals
    stresses = stress(state, params, fspace)
    cell_inputs = np.hstack((stresses.reshape(-1, 1), state.chemical)) 
    divrate = nn_fun_t.apply(params["nn"], state.key, cell_inputs).reshape(-1,)
    # For my case, the divrates HAVE to depend on field - the learned parameters
    # will counteract this field (unsure how to make this more general)
    max_divrate = logistic(state.field, 0.1, 25.0)
    divrate = np.multiply(max_divrate, divrate)
    # divrate is zero if celltype is zero
    divrate = np.where(state.celltype>0, divrate, 0.0)
    divrate = divrate*logistic(state.radius+.06, 50, params['cellRad'])
    return divrate

def S_set_divrate(state, params, fspace, divrate_fn=div_mechanical,**kwargs):
    """ Sets divrates."""
    divrate = divrate_fn(state, params, fspace, **kwargs)
    new_state = dataclasses.replace(state, divrate=divrate)
    
    return new_state