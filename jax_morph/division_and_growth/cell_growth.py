import jax.numpy as np
import jax_md.dataclasses as jdc


def S_grow_cells(state, params, fspace=None, grate=.02, type='linear'):
    
    #constant growth
    if type=='linear':
        new_radius = state.radius + grate
    elif type=='exponential':
        new_radius = state.radius * np.exp(grate)
    
    #set max radius
    new_radius = np.where(new_radius<params['cellRad'], new_radius, params['cellRad'])
    
    new_state = jdc.replace(state, radius=new_radius)
    
    return new_state