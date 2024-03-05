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
    
    def return_nbrs(self) -> bool:
        return True
        
    def _sgd(self, state, nbrs, energy_fn):
        
        init, apply = jax_md.minimize.gradient_descent(energy_fn, state.shift, self.dt) 

    
        def scan_fn(opt_state, i):
            state, nbrs = opt_state
            state = apply(state, neighbor=nbrs)
            nbrs = nbrs.update(state)
            return (state, nbrs), 0.

        #relax system
        opt_state = (init(state.position), nbrs)
        opt_state, _ = jax.lax.scan(scan_fn, opt_state, np.arange(self.relaxation_steps))
        opt_state, nbrs = opt_state
        
        return opt_state, nbrs
    
    
    @jax.named_scope("jax_morph.SGDMechanicalRelaxation")
    def __call__(self, state, nbrs, *, key=None, **kwargs):

        #generate morse pair potential
        energy_fn = self.mechanical_potential.energy_fn(state)
        
        #minimize
        new_positions, new_nbrs = self._sgd(state, nbrs, energy_fn)

        state = eqx.tree_at(lambda s: s.position, state, new_positions)

        return state, new_nbrs

class BrownianMechanicalRelaxation(SimulationStep):
    mechanical_potential:   eqx.Module
    #relaxation_steps:       int = eqx.field(default=25, static=True)
    relaxation_steps:       int = eqx.field(default=100, static=True)
    dt:                     float = eqx.field(default=8e-4, static=True)
    kT:                     float = eqx.field(default=1., static=True)


    def return_logprob(self) -> bool:
        return False
    
    def return_nbrs(self) -> bool:
        return True
        
    def _sgd(self, state, nbrs, energy_fn, key):
        init, apply = jax_md.simulate.brownian(energy_fn, state.shift, dt=self.dt, kT=self.kT, gamma=.8)
        #init, apply = jax_md.minimize.gradient_descent(energy_fn, state.shift, self.dt) 
        apply_fn = eqx.filter_jit(apply)
    
        def scan_fn(opt_state, i):
            state, nbrs = opt_state
            nbrs = nbrs.update(state.position)
            state = apply_fn(jax.lax.stop_gradient(state), neighbor=nbrs)
            return (state, nbrs), 0.

        #relax system
        opt_state = (init(key, state.position), nbrs)
        opt_state, _ = jax.lax.scan(scan_fn, opt_state, np.arange(self.relaxation_steps))
        opt_state, nbrs = opt_state
        
        return opt_state.position, nbrs
    
    
    @jax.named_scope("jax_morph.BrownianMechanicalRelaxation")
    def __call__(self, state, nbrs, *, key=None, **kwargs):

        #generate morse pair potential
        energy_fn = self.mechanical_potential.energy_fn(state)
        
        #minimize
        new_positions, new_nbrs = self._sgd(state, nbrs, energy_fn, key)

        state = eqx.tree_at(lambda s: s.position, state, new_positions)

        return state, new_nbrs