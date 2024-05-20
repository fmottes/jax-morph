import jax
import jax.numpy as np

import jax_md

import equinox as eqx

from ..._base import SimulationStep
from ...utils import discount_tangent

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


# Brownian Relaxation with discounted gradient and dynamic kT
# default behavior should be no discounting, but state will have
# to have kT

class BrownianMechanicalRelaxation(SimulationStep):
    mechanical_potential:   eqx.Module
    relaxation_steps:       int = eqx.field(default=1, static=True)
    dt:                     float = eqx.field(default=8e-4, static=True)
    kT:                     float = eqx.field(default=1., static=True)
    gamma:                  float = eqx.field(default=.8, static=True)
    discount:               float = eqx.field(default=1., static=True)


    def return_logprob(self) -> bool:
        return False
        
    def _brownian(self, state, energy_fn, kT, gamma, key):
        
        init, apply = jax_md.simulate.brownian(energy_fn, state.shift, dt=self.dt, kT=kT, gamma=gamma)

        def scan_fn(opt_state, i):
            opt_state = apply(opt_state)
            opt_state = discount_tangent(opt_state, self.discount)
            return opt_state, 0.

        #relax system
        opt_state = init(key, state.position)
        opt_state, _ = jax.lax.scan(scan_fn, opt_state, np.arange(self.relaxation_steps))
        
        return opt_state.position
    
    
    @jax.named_scope("jax_morph.BrownianMechanicalRelaxation")
    def __call__(self, state, *, key=None, **kwargs):

        _kT = state.kT if hasattr(state, 'kT') else self.kT
        _gamma = state.gamma if hasattr(state, 'gamma') else self.gamma

        #generate morse pair potential
        energy_fn = self.mechanical_potential.energy_fn(state)
        
        #minimize
        new_positions = self._brownian(state, energy_fn, _kT, _gamma, key)

        state = eqx.tree_at(lambda s: s.position, state, new_positions)

        return state