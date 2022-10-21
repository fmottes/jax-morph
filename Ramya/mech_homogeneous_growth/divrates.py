import jax.numpy as np
from jax import jit, lax, vmap

import jax_md.dataclasses as jax_dataclasses

from jax_morph.utils import logistic
from jax_morph.datastructures import CellState

def div_mechanical(state: CellState,
                params: dict,
                ) -> np.array:

    div_gamma = params['div_gamma']
    div_k = params['div_k']

    ### NOTE: possibility to extend this function to account for more complex interactions
    # Calculate stresses
    
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