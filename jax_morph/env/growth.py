import jax
import jax.numpy as np

import equinox as eqx

from ..simulation import SimulationStep





class CellGrowth(SimulationStep):
    max_radius:     float
    growth_rate:    float
    growth_type:    str = eqx.field(static=True)


    def return_logprob(self) -> bool:
        return False
    
    
    def __init__(self, *, 
                 growth_rate=1., 
                 max_radius=.5, 
                 growth_type='linear',
                 **kwargs
                 ):

        # if not hasattr(state, 'radius'):
        #     raise AttributeError('CellState must have "radius" attribute')
        
        if growth_type not in ['linear', 'exponential']:
            raise ValueError('growth_type must be either "linear" or "exponential"')
        
        self.growth_rate = growth_rate
        self.max_radius = max_radius
        self.growth_type = growth_type



    @jax.named_scope("jax_morph.CellGrowth")
    def __call__(self, state, *, key=None, **kwargs):

        if self.growth_type == 'linear':
            new_radius = state.radius + self.growth_rate
        elif self.growth_type == 'exponential':
            new_radius = state.radius*np.exp(self.growth_rate)

        new_radius = np.where(new_radius > self.max_radius, self.max_radius, new_radius)*np.where(state.celltype.sum(1)[:,None]>0, 1, 0)

        state = eqx.tree_at(lambda s: s.radius, state, new_radius)

        return state