import jax
import jax.numpy as np

import jax_md.dataclasses as jdc

import haiku as hk
import equinox as eqx

from jax_morph.utils import differentiable_clip

def _standardize(x):
    #numpy std spits errors when taking gradients
    m = np.mean(x)
    std = np.sqrt(np.mean((x - m)**2)+1e-10)
    return (x - m) / std  #, axis=1, keepdims=True

def _normalize(x):
    return x / (np.sqrt(np.sum(x**2, axis=1, keepdims=True))+1e-10)


def hidden_state_nn(params, 
           train_params=None, 
           n_hidden=3,
           use_state_fields=None,
           train=True,
           transform_mlp_out=None,
          ):
    
    if use_state_fields is None:
        raise ValueError('Input fields flags must be passed explicitly as a CellState dataclass.')
    
    if type(n_hidden) == np.int_ or type(n_hidden) == int:
        n_hidden = [int(n_hidden)]

    if transform_mlp_out is None:
        transform_mlp_out = lambda x: x

    def _hidden_nn(in_fields):
        mlp = hk.nets.MLP(n_hidden+[params['hidden_state_size']],
                          activation=jax.nn.leaky_relu,
                          w_init=hk.initializers.Orthogonal(),
                          activate_final=False
                         )
        
        out = mlp(in_fields)
        out = transform_mlp_out(out)

        return out

    _hidden_nn = hk.without_apply_rng(hk.transform(_hidden_nn))


    
    def init(state, key):     
        
        in_fields = np.hstack([f if len(f.shape)>1 else f[:,np.newaxis] for f in jax.tree_leaves(eqx.filter(state, use_state_fields))])
        
        input_dim = in_fields.shape[1]
            
        p = _hidden_nn.init(key, np.ones(input_dim))
        
        #add to param dict
        params['hidden_fn'] = p
        
        
        # no need to update train_params when generating initial state
        if type(train_params) is dict:
            
            #set trainability flag
            train_p = jax.tree_map(lambda x: train, p)

            train_params['hidden_fn'] = train_p
            
            return params, train_params
        
        else:
            return params
            
        
        
    def fwd(state, params):    

        in_fields = np.hstack([f if len(f.shape)>1 else f[:,np.newaxis] for f in jax.tree_leaves(eqx.filter(state, use_state_fields))])
        #in_fields = np.hstack([_normalize(f) if len(f.shape)>1 else _normalize(f[:,np.newaxis]) for f in jax.tree_leaves(eqx.filter(state, use_state_fields))])
        #in_fields = _standardize(in_fields)

        delta_hidden_state = _hidden_nn.apply(params['hidden_fn'], in_fields)
    
        return delta_hidden_state
    
    
    return init, fwd




# STATE UPDATE FUNCTION
def S_hidden_state(state, params, fspace=None, dhidden_fn=None, state_decay=.8):
    
    if None == dhidden_fn:
        raise(ValueError('Need to pass a valid function for the calculation of the new hidden state.'))

    #regulation = state_decay*state.regulation + dhidden_fn(state, params)
    #regulation = regulation / np.sum(np.abs(regulation), axis=1, keepdims=True) 

    # regulation = dhidden_fn(state, params)
    # hidden_state = jax.nn.sigmoid(regulation)
        
    # normalize hidden state
    #hidden_state = hidden_state / np.sum(np.abs(hidden_state), axis=1, keepdims=True)

    # state = jdc.replace(state, regulation=regulation, hidden_state=hidden_state)


    if state_decay > 0.:
        hidden_state = state_decay*state.hidden_state + (1-state_decay)*dhidden_fn(state, params)  #(1-state_decay)*dhidden_fn(state, params)
    else:
        hidden_state = dhidden_fn(state, params)

    hidden_state = differentiable_clip(hidden_state, -1e2, 1e2)

    state = jdc.replace(state, hidden_state=hidden_state)
    
    return state