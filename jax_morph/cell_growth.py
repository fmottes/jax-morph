import jax.numpy as np
import jax_md.dataclasses as jax_dataclasses


def S_grow_cells(state, params, fspace=None, grate=.1):
    
    #constant growth
    new_radius = state.radius * np.exp(grate)
    
    #set max radius - try without for now
    new_radius = np.where(new_radius<params['cellRad'], new_radius, params['cellRad'])
    
    new_state = jax_dataclasses.replace(state, radius=new_radius)
    
    return new_state