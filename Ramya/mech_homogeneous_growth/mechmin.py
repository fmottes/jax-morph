import jax.numpy as np
from jax import jit, lax, vmap

from jax_md import minimize
import jax_md.dataclasses as jax_dataclasses



def mechmin_sgd(energy, state, params, fspace, dt=.001, nbrs=None):
    '''
    Generic function for the mechanical relaxation of the system with SGD.
    '''
    
    init, apply = minimize.gradient_descent(energy, fspace.shift, dt) # 0.001 is a timestep that seems to work.
    #apply = jit(apply)
 
    #@jit
    def scan_fn(opt_state, i):
        return apply(opt_state), 0.

    #initialize
    opt_state = init(state.position)
    #update
    n_steps = params['mech_relaxation_steps']
    opt_state, _ = lax.scan(scan_fn, opt_state, np.arange(n_steps))
    
    return opt_state