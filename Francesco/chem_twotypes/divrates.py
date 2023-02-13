import haiku as hk
import equinox as eqx

import jax
import jax.numpy as np
from jax import jit, lax, vmap

import jax_md.dataclasses as jax_dataclasses

from jax_morph.utils import logistic
from jax_morph.datastructures import CellState


def S_set_divrate(state, params, fspace=None, divrate_fn=None):
    
    if None == divrate_fn:
        raise(ValueError('Need to pass a valid function for the calculation of the division rates.'))

    divrate = divrate_fn(state, params)
    
    new_state = jax_dataclasses.replace(state, divrate=divrate)
    
    return new_state



# GENERATE DIVISION FUNCTION WITH NEURAL NETWORK
def div_nn(params, 
           train_params=None, 
           n_hidden=3,
           use_state_fields=CellState(*tuple([False]*3+[True]*2+[False]*3)),
           train=True,
          ):
    
    if type(n_hidden) == np.int_ or type(n_hidden) == int:
        n_hidden = [int(n_hidden)]
    

    
    def _div_nn(in_fields):
        mlp = hk.nets.MLP(n_hidden+[1],
                          activation=jax.nn.leaky_relu,
                          activate_final=False
                         )
        #out = jax.nn.softplus(mlp(in_fields))
        out = jax.nn.sigmoid(mlp(in_fields))
        return out

    _div_nn = hk.without_apply_rng(hk.transform(_div_nn))


    
    def init(state, key):
        
        
        in_fields = np.hstack([f if len(f.shape)>1 else f[:,np.newaxis] for f in jax.tree_leaves(eqx.filter(state, use_state_fields))])
        
        input_dim = in_fields.shape[1]
            
        p = _div_nn.init(key, np.zeros(input_dim))
        
        #add to param dict
        params['div_fn'] = p
        
        # no need to update train_params when generating initial state
        if type(train_params) is dict:
            
            #set trainability flag
            train_p = jax.tree_map(lambda x: train, p)

            train_params['div_fn'] = train_p
        
            return params, train_params
            
        else:
            return params
            
        
        
    def fwd(state, params):
        
        in_fields = np.hstack([f if len(f.shape)>1 else f[:,np.newaxis] for f in jax.tree_leaves(eqx.filter(state, use_state_fields))])
        
        x = _div_nn.apply(params['div_fn'], in_fields).flatten()
        
        divrate = x*logistic(state.radius+.06, 50, params['cellRad'])
        
        divrate = np.where(state.celltype<1.,0,divrate)
    
        return divrate
    
    
    return init, fwd




#CHEMICAL DIVISION RATES BASED ON LOGISTICS

#DANGER: change of conventions for chemicals (wrt Alma's code)!!!!
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


