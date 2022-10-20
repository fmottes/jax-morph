import jax.numpy as np
from jax import jit, lax, vmap

import jax_md.dataclasses as jax_dataclasses

from jax_morph.utils import logistic
from jax_morph.datastructures import CellState



#DANGER: change of conventions for chemicals!!!!
#Now 
#species 0 produces chemical 0 and divides according to chemical 1
#species 1 produces chemical 1 and divides according to chemical 0


def div_chemical(state: CellState,
                params: dict,
                ) -> np.array:

    div_gamma = params['div_gamma']
    div_k = params['div_k']

    ### NOTE: possibility to extend this function to account for more complex interactions
    
    #calculate "rates"
    div1 = logistic(state.chemical[:,1],div_gamma[0],div_k[0])
    div2 = logistic(state.chemical[:,0],div_gamma[1],div_k[1])
    
    #create array with new divrates
    divrate = np.where(state.celltype==1,div1,div2)
    divrate = np.where(state.celltype==0,0,divrate)
    
    #cells cannot divide if they are too small
    #constants are arbitrary, change if you change cell radius
    divrate = divrate*logistic(state.radius+.06, 50, params['cellRad'])
    
    return divrate


def S_set_divrate(state, params, fspace=None):
    
    divrate = div_chemical(state,params)
    
    new_state = jax_dataclasses.replace(state, divrate=divrate)
    
    return new_state