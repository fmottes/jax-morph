import jax
import jax.numpy as np
import jax.tree_util as jtu

import equinox as eqx

from ..simulation import simulate

from functools import partial



@eqx.filter_jit
def reinforce_loss(model, 
                   istate, 
                   *, 
                   cost_fn, 
                   key, 
                   n_steps=1, 
                   BATCH_SIZE=1, 
                   GAMMA=.9, 
                   LAMBDA_L1=0.,
                   normalize_cost=True,
                   **kwargs
                   ):
    
    
    if istate.celltype.shape[0] - istate.celltype.sum() < n_steps:
        raise ValueError('n_steps must be smaller than the number of available empty cell slots in the initial state. Please elongate the initial state with istate.elongate() or decrease the number of steps.')
    

    subkeys = jax.random.split(key, BATCH_SIZE)
    subkeys = np.asarray(subkeys)

    vsim = jax.vmap(partial(simulate, history=True), (None, None, 0, None))
    trajectory, logp = vsim(model, istate, subkeys, n_steps)


    istate = jtu.tree_map(lambda x: np.repeat(x[None,None,:,:],BATCH_SIZE,0), istate)

    trajectory = jtu.tree_map(lambda *v: np.concatenate(v,1), *[istate, trajectory])

    cost = jax.vmap(cost_fn)(trajectory)


    #discounted costs
    def _returns(costs):
        def _scan_fn(G, c):
            G = G*GAMMA + c
            return G, G
        _, Gs = jax.lax.scan(_scan_fn, np.asarray(0.), costs[::-1])
        return Gs[::-1]
    
    
    if GAMMA > 0.:
        cost = jax.vmap(_returns)(cost)

    #flatten before normalization (per batch) but after discounting (per episode)
    cost = cost.flatten()

    if normalize_cost:
        cost = (cost-cost.mean(-1, keepdims=True))/(cost.std(-1, keepdims=True)+1e-8)


    #no - sign because we assume a cost instead of a reward
    loss = np.sum(jax.lax.stop_gradient(cost)*logp.flatten())


    #L1 penalty on weights
    if LAMBDA_L1 > 0.:
        reg = jax.tree_map(lambda x: np.abs(x).sum(), eqx.filter(model, eqx.is_array))
        reg = LAMBDA_L1 * jax.tree_util.tree_reduce(lambda x,y: x+y, reg)
        return loss + reg
    
    else:
        return loss