import jax
import jax.numpy as np

import equinox as eqx

from ..._base import SimulationStep

from typing import Union, Sequence, Callable


###-----------DIVISION MLP-----------------###

class DivisionMLP(SimulationStep):
    input_fields:        Sequence[str] = eqx.field(static=True)
    transform_output:    Union[Callable,None] = eqx.field(static=True)
    mlp:                 eqx.nn.MLP


    def return_logprob(self) -> bool:
        return False


    def __init__(self, state, 
                 input_fields, 
                 *,
                 key,
                 w_init=jax.nn.initializers.constant(0.),
                 transform_output=None,
                 final_activation=jax.nn.softplus,
                 **kwargs
                 ):

        self.transform_output = transform_output
        self.input_fields = input_fields

        in_shape = np.concatenate([getattr(state, field) for field in input_fields], axis=1).shape[-1]
        out_shape = 1

        self.mlp = eqx.nn.MLP(in_size=in_shape,
                            out_size=out_shape,
                            depth=0,
                            width_size=0,
                            activation=jax.nn.leaky_relu,
                            final_activation=final_activation,
                            key=key,
                            **kwargs
                            )
        
        self.mlp = eqx.tree_at(lambda m: m.layers[0].weight, self.mlp, w_init(key, self.mlp.layers[0].weight.shape))


    @jax.named_scope("jax_morph.DivisionMLP")
    def __call__(self, state, *, key=None, **kwargs):

        #concatenate input features
        in_features = np.concatenate([getattr(state, field) for field in self.input_fields], axis=1)


        #apply MLP
        division = jax.vmap(self.mlp)(in_features)


        #transform output
        if self.transform_output is not None:
            division = self.transform_output(state, division)
        

        #keep only alive cells
        division = np.where(state.celltype.sum(1)[:,None] > 0, division, 0.)

        #update state
        state = eqx.tree_at(lambda s: s.division, state, division)

        return state
    



###-----------SECRETION MLP-----------------###

class SecretionMLP(SimulationStep):
    input_fields:        Sequence[str] = eqx.field(static=True)
    transform_output:    Union[Callable,None] = eqx.field(static=True)
    ctype_sec_chem:      eqx.field(static=True)
    mlp:                 eqx.nn.MLP


    def return_logprob(self) -> bool:
        return False


    def __init__(self, 
                state, 
                input_fields,
                *,
                key,
                ctype_sec_chem=None,
                w_init=jax.nn.initializers.orthogonal(),
                transform_output=None,
                final_activation=lambda x: jax.nn.sigmoid(x/10.),
                **kwargs
                ):

        self.transform_output = transform_output
        self.input_fields = input_fields

        in_shape = np.concatenate([getattr(state, field) for field in input_fields], axis=1).shape[1]
        out_shape = state.chemical.shape[-1]

        self.mlp = eqx.nn.MLP(in_size=in_shape,
                            out_size=out_shape,
                            depth=0,
                            width_size=0,
                            activation=jax.nn.leaky_relu,
                            final_activation=final_activation,
                            key=key,
                            **kwargs
                            )
        
        self.mlp = eqx.tree_at(lambda m: m.layers[0].weight, self.mlp, w_init(key, self.mlp.layers[0].weight.shape))


        if ctype_sec_chem is None:
            self.ctype_sec_chem = np.repeat(np.atleast_2d([1.]*state.chemical.shape[1]), state.celltype.shape[-1], axis=0).tolist()
            
        else:
            if np.asarray(ctype_sec_chem).shape != (state.celltype.shape[1], state.chemical.shape[1]):
                raise ValueError("ctype_sec_chem must be shape (N_CELLTYPE, N_CHEM)")
            
            self.ctype_sec_chem = ctype_sec_chem



    @jax.named_scope("jax_morph.SecretionMLP")
    def __call__(self, state, *, key=None, **kwargs):

        #concatenate input features
        in_features = np.concatenate([getattr(state, field) for field in self.input_fields], axis=1)


        #apply MLP
        secretion_rate = jax.vmap(self.mlp)(in_features)

        #transform output
        if self.transform_output is not None:
            secretion_rate = self.transform_output(state, secretion_rate)


        sec_mask = state.celltype @ np.atleast_2d(self.ctype_sec_chem)

        secretion_rate = sec_mask*secretion_rate


        #update state
        state = eqx.tree_at(lambda s: s.secretion_rate, state, secretion_rate)

        return state
    




###-----------HIDDEN STATE MLP-----------------###

class HiddenStateMLP(SimulationStep):
    input_fields:        Sequence[str] = eqx.field(static=True)
    transform_output:    Union[Callable,None] = eqx.field(static=True)
    mlp:                 eqx.nn.MLP
    memory_decay:        Union[float, jax.Array]


    def return_logprob(self) -> bool:
        return False


    def __init__(self, 
                state, 
                input_fields,
                *,
                key,
                layer_width=128,
                num_mlp_hidden_layers=1,
                memory_decay=.7,
                transform_output=None,
                final_activation=lambda x: x,
                **kwargs
                ):

        self.memory_decay = memory_decay
        self.transform_output = transform_output
        self.input_fields = input_fields

        in_shape = np.concatenate([getattr(state, field) for field in input_fields], axis=1).shape[-1]
        out_shape = state.hidden_state.shape[1]

        self.mlp = eqx.nn.MLP(in_size=in_shape,
                            out_size=out_shape,
                            depth=int(num_mlp_hidden_layers+1),
                            width_size=int(layer_width),
                            activation=jax.nn.leaky_relu,
                            final_activation=final_activation,
                            key=key,
                            **kwargs
                            )
        



    @jax.named_scope("jax_morph.HiddenStateMLP")
    def __call__(self, state, *, key=None, **kwargs):

        #concatenate input features
        in_features = np.concatenate([getattr(state, field) for field in self.input_fields], axis=1)

        #apply MLP
        delta_hs = jax.vmap(self.mlp)(in_features)


        #transform output
        if self.transform_output is not None:
            delta_hs = self.transform_output(state, delta_hs)

        #keep only alive cells
        delta_hs = np.where(state.celltype.sum(1)[:,None] > 0, delta_hs, 0.)

        hidden_state = self.memory_decay*state.hidden_state + (1-self.memory_decay)*delta_hs

        #update state
        state = eqx.tree_at(lambda s: s.hidden_state, state, hidden_state)

        return state