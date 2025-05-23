from typing import Any
import jax
import jax.numpy as np
import jax.tree_util as jtu

import equinox as eqx

from ..simulation import simulate

from functools import partial
from collections import namedtuple


# every loss function should return a Loss object
Loss = namedtuple('Loss', ['loss_fn', 'has_aux'])


def ReinforceLoss(cost_fn, *, n_sim_steps, n_episodes=1, n_val_episodes=0, gamma=.9, lambda_l1=0., normalize_cost_returns='episode', reg_f=None):

    n_episodes = int(n_episodes)
    n_val_episodes = int(n_val_episodes)
    gamma = float(gamma)
    lambda_l1 = float(lambda_l1)
    n_sim_steps = int(n_sim_steps)

    if (normalize_cost_returns not in ['batch', 'episode']) and (normalize_cost_returns is not False):
        raise ValueError("normalize_cost_returns must be 'batch', 'episode' or False, got {}".format(normalize_cost_returns))


    def _reinforce_loss(model, istate, *, key, n_sim_steps=n_sim_steps, n_val_episodes=n_val_episodes, **kwargs):

        vsim = jax.vmap(partial(simulate, history=True), (None, None, 0, None))


        key, *subkeys = jax.random.split(key, n_episodes+1)
        trajectory, logp = vsim(model, istate, np.asarray(subkeys), n_sim_steps)


        #add istate to beginning of trajectory
        _istate = jtu.tree_map(lambda x: np.repeat(x[None,None,:,:],n_episodes,0), istate)
        trajectory = jtu.tree_map(lambda *v: np.concatenate(v,1), *[_istate, trajectory])

        cost = jax.vmap(cost_fn)(trajectory)


        #calculate actual cost for validation
        if n_val_episodes > 0:

            n_val_episodes = int(n_val_episodes)

            key, *subkeys = jax.random.split(key, n_val_episodes+1)
            val_trajectory, _ = vsim(model, istate, np.asarray(subkeys), n_sim_steps)

            #add istate to beginning of val_trajectory
            _istate = jtu.tree_map(lambda x: np.repeat(x[None,None,:,:],n_val_episodes,0), istate)
            val_trajectory = jtu.tree_map(lambda *v: np.concatenate(v,1), *[_istate, val_trajectory])

            val_cost = jax.vmap(cost_fn)(val_trajectory).sum(-1).mean()



        #discounted costs
        def _returns(costs):
            def _scan_fn(G, c):
                G = G*gamma + c
                return G, G
            _, Gs = jax.lax.scan(_scan_fn, np.asarray(0.), costs[::-1])
            return Gs[::-1]
        
        
        if gamma > 0.:
            cost = jax.vmap(_returns)(cost)

        

        if 'batch' == normalize_cost_returns:
            #flatten before normalization (per batch) but after discounting (per episode)
            cost = cost.flatten()
            cost = (cost-cost.mean(-1, keepdims=True))/(cost.std(-1, keepdims=True)+1e-8)

        elif 'episode' == normalize_cost_returns:
            #normalize cost per episode
            cost = (cost-cost.mean(-1, keepdims=True))/(cost.std(-1, keepdims=True)+1e-8)
            cost = cost.flatten()
        else:
            cost = cost.flatten()


        #no - sign because we assume a cost instead of a reward
        loss = np.sum(jax.lax.stop_gradient(cost)*logp.flatten())


        #L1 penalty on weights
        if lambda_l1 > 0.:
            reg = jax.tree_map(lambda x: np.abs(x).sum(), eqx.filter(model, eqx.is_array))
            reg = lambda_l1 * jax.tree_util.tree_reduce(lambda x,y: x+y, reg)
            loss = loss + reg

        if reg_f is not None:
            loss = loss + reg_f(model)
        

        if n_val_episodes > 0:
            return loss, val_cost
        else:
            return loss

    return Loss(loss_fn=_reinforce_loss, has_aux=(n_val_episodes > 0))
    
def SimpleLoss(cost_fn, *, n_sim_steps, n_episodes=1, n_val_episodes=0, lambda_l1=0., normalize_cost_returns=False, istate_func=None, checkpoint=False):

    n_episodes = int(n_episodes)
    n_val_episodes = int(n_val_episodes)
    lambda_l1 = float(lambda_l1)
    n_sim_steps = int(n_sim_steps)
    if istate_func is None:
        istate_func = lambda k, i: i
        
    if (normalize_cost_returns not in ['batch', 'episode']) and (normalize_cost_returns is not False):
        raise ValueError("normalize_cost_returns must be 'batch', 'episode' or False, got {}".format(normalize_cost_returns))


    def _simple_loss(model, istate, *, key, n_sim_steps=n_sim_steps, n_val_episodes=n_val_episodes, **kwargs):

        def _sim(key, istate, model, n_sim_steps):
            istate = istate_func(key, istate)
            trajectory = simulate(model, istate, key, n_sim_steps, history=True, checkpoint=checkpoint)
            
            # My simulations don't have cell divisions so no logprob returned.
            if isinstance(trajectory, tuple):
                trajectory = trajectory[0]    
                
            _istate = jtu.tree_map(lambda x: x[None,:, :], istate)
            trajectory = jtu.tree_map(lambda *v: np.concatenate(v), *[_istate, trajectory])
            return trajectory
            
        vsim = jax.vmap(_sim, (0, None, None, None))
        key, *subkeys = jax.random.split(key, n_episodes+1)
        trajectory = vsim(np.asarray(subkeys), istate, model, n_sim_steps)

        
        cost = jax.vmap(cost_fn)(trajectory)
 
        #calculate actual cost for validation
        if n_val_episodes > 0:

            n_val_episodes = int(n_val_episodes)

            key, *subkeys = jax.random.split(key, n_val_episodes+1)
            val_trajectory = vsim(np.asarray(subkeys), istate, model, n_sim_steps)

            if isinstance(val_trajectory, tuple):
                val_trajectory = val_trajectory[0]


            val_cost = jax.vmap(cost_fn)(val_trajectory).sum(-1).mean()
        

        if 'batch' == normalize_cost_returns:
            #flatten before normalization (per batch) but after discounting (per episode)
            cost = cost.flatten()
            cost = (cost-cost.mean(-1, keepdims=True))/(cost.std(-1, keepdims=True)+1e-8)

        elif 'episode' == normalize_cost_returns:
            #normalize cost per episode
            cost = (cost-cost.mean(-1, keepdims=True))/(cost.std(-1, keepdims=True)+1e-8)
            cost = cost.flatten()
        else:
            cost = cost.flatten()


        #loss = cost.sum(0).mean()
        loss = cost.mean()


        #L1 penalty on weights
        if lambda_l1 > 0.:
            reg = jax.tree_map(lambda x: np.abs(x).sum(), eqx.filter(model, eqx.is_array))
            reg = lambda_l1 * jax.tree_util.tree_reduce(lambda x,y: x+y, reg)
            loss = loss + reg
        

        if n_val_episodes > 0:
            return loss, val_cost
        else:
            return loss

    return Loss(loss_fn=_simple_loss, has_aux=(n_val_episodes > 0))
