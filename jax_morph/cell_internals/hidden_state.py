import jax
import jax.numpy as np

import jax_md.dataclasses as jdc

import haiku as hk
import equinox as eqx



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
        
        delta_hidden_state = _hidden_nn.apply(params['hidden_fn'], in_fields)
    
        return delta_hidden_state
    
    
    return init, fwd




# STATE UPDATE FUNCTION
def S_hidden_state(state, params, fspace=None, dhidden_fn=None, state_decay=.9):
    
    if None == dhidden_fn:
        raise(ValueError('Need to pass a valid function for the calculation of the new hidden state.'))


    # hidden_state = state.hidden_state
    # hidden_state = np.log(hidden_state/(1-hidden_state)+1e-5)
    # hidden_state = hidden_state + hidden_fn(state, params)
    # hidden_state = jax.nn.sigmoid(hidden_state)

    # normalize hidden state ?

    regulation = state_decay*state.regulation + dhidden_fn(state, params)

    hidden_state = jax.nn.sigmoid(regulation)
    
    new_state = jdc.replace(state, regulation=regulation, hidden_state=hidden_state)
    
    return new_state