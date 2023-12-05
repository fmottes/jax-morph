import jax
import jax.numpy as np

import jax_md

import equinox as eqx

from ..._base import SimulationStep


class SGDMechanicalRelaxation(SimulationStep):
    mechanical_potential:   eqx.Module
    relaxation_steps:       int = eqx.field(default=25, static=True)
    dt:                     float = eqx.field(default=8e-4, static=True)


    def return_logprob(self) -> bool:
        return False


    def _sgd(self, state, pair_potential):

        init, apply = jax_md.minimize.gradient_descent(pair_potential, state.shift, self.dt) 
 
        def scan_fn(opt_state, i):
            return apply(opt_state), 0.

        #relax system
        opt_state = init(state.position)
        opt_state, _ = jax.lax.scan(scan_fn, opt_state, np.arange(self.relaxation_steps))

        return opt_state
    
    
    @jax.named_scope("jax_morph.SGDMechanicalRelaxation")
    def __call__(self, state, *, key=None, **kwargs):

        #generate morse pair potential
        energy_fn = self.mechanical_potential.energy_fn(state)
        
        #minimize
        new_positions = self._sgd(state, energy_fn)

        state = eqx.tree_at(lambda s: s.position, state, new_positions)

        return state