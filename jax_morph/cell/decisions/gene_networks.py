import jax
import jax.numpy as np

import jax_md

import equinox as eqx

from ...simulation import SimulationStep

from typing import Union, Sequence, Callable



class GeneNetwork(SimulationStep):
    input_fields:        Sequence[str] = eqx.field(static=True)
    output_fields:       Sequence[str] = eqx.field(static=True)
    out_indices:         tuple = eqx.field(static=True)
    transform_output:    Union[Callable,None] = eqx.field(static=True)
    n_solver_steps:      int = eqx.field(static=True)
    dt:                  float = eqx.field(static=True)

    interaction_matrix:  jax.Array
    degradation_rate:    Union[float, jax.Array]
    expr_level_decay:    Union[float, jax.Array]


    def return_logprob(self) -> bool:
        return False
    

    def circuit_solve(self, x0, I):
    
        def _step(xt,t):
            xtt = xt + self.x_dot(xt,I)*self.dt
            return xtt, 0.

        x, _ = jax.lax.scan(_step, x0, np.arange(self.n_solver_steps))
        
        return x
    

    def x_dot(self, xt, I):
        return jax.nn.sigmoid(xt @ self.interaction_matrix) - self.degradation_rate * xt + I



    def __init__(self, 
                state, 
                input_fields,
                output_fields,
                *,
                key,
                expr_level_decay=.7,
                interaction_init=jax.nn.initializers.normal(1.),
                degradation_init=jax.nn.initializers.constant(1.),
                transform_output=None,
                n_solver_steps=int(1e2),
                dt=.1,
                **kwargs
                ):
        


        self.input_fields = input_fields
        self.output_fields = output_fields
        self.expr_level_decay = float(expr_level_decay)

        self.n_solver_steps = int(n_solver_steps)
        self.dt = dt

        in_shape = np.concatenate([getattr(state, field) for field in input_fields], axis=1).shape[-1]
        out_shape = np.concatenate([getattr(state, field) for field in output_fields], axis=1).shape[-1]

        system_size = int(in_shape + state.hidden_state.shape[-1] + out_shape)

        self.interaction_matrix = interaction_init(key, shape=(system_size, system_size))
        self.degradation_rate = degradation_init(key, shape=(1, system_size))

        out_sizes = [getattr(state, field).shape[-1] for field in self.output_fields]
        self.out_indices = tuple((system_size - np.cumsum(np.asarray(out_sizes)[::-1])).tolist()[::-1] + [system_size])

        
        if transform_output is None:
            self.transform_output = None
        else:
            self.transform_output = dict(zip(self.output_fields, [None]*len(self.output_fields)))
            self.transform_output.update(transform_output)



    @jax.named_scope("jax_morph.GeneNetwork")
    def __call__(self, state, *, key=None, **kwargs):

        #concatenate input features
        in_features = np.concatenate([getattr(state, field) for field in self.input_fields], axis=1)
        out_features = np.concatenate([getattr(state, field) for field in self.output_fields], axis=1)

        gene_state = np.concatenate([in_features, self.expr_level_decay*state.hidden_state, out_features], axis=1)
        I = np.concatenate([in_features, np.zeros_like(state.hidden_state), np.zeros_like(out_features)], axis=1)

        alive = np.where(state.celltype.sum(1)>0., 1., 0.)[:,None]

        gene_state = self.circuit_solve(gene_state, I) * alive

        hidden_state = gene_state[:, in_features.shape[-1]:in_features.shape[-1]+state.hidden_state.shape[-1]]

        #update state
        state = eqx.tree_at(lambda s: s.hidden_state, state, hidden_state)

        #update output
        for i, field in enumerate(self.output_fields):

            new_field = gene_state[:, self.out_indices[i]:self.out_indices[i+1]]

            if self.transform_output[field] is not None:
                new_field = self.transform_output[field](state, new_field) * alive

            state = eqx.tree_at(lambda s: getattr(s, field), state, new_field)

        return state