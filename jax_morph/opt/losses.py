from typing import Any
import jax
import jax.numpy as np
import jax.tree_util as jtu

import equinox as eqx

from ..simulation import simulate

from functools import partial



def ReinforceLoss(cost_fn, *, n_sim_steps, n_episodes=1, gamma=.9, lambda_l1=0., normalize_cost_returns=True):

    n_episodes = int(n_episodes)
    gamma = float(gamma)
    lambda_l1 = float(lambda_l1)
    normalize_cost_returns = bool(normalize_cost_returns)
    n_sim_steps = int(n_sim_steps)


    def _reinforce_loss(model, istate, *, key, n_sim_steps=n_sim_steps, **kwargs):

        subkeys = jax.random.split(key, n_episodes)
        subkeys = np.asarray(subkeys)

        vsim = jax.vmap(partial(simulate, history=True), (None, None, 0, None))
        trajectory, logp = vsim(model, istate, subkeys, n_sim_steps)


        istate = jtu.tree_map(lambda x: np.repeat(x[None,None,:,:],n_episodes,0), istate)

        trajectory = jtu.tree_map(lambda *v: np.concatenate(v,1), *[istate, trajectory])

        cost = jax.vmap(cost_fn)(trajectory)


        #discounted costs
        def _returns(costs):
            def _scan_fn(G, c):
                G = G*gamma + c
                return G, G
            _, Gs = jax.lax.scan(_scan_fn, np.asarray(0.), costs[::-1])
            return Gs[::-1]
        
        
        if gamma > 0.:
            cost = jax.vmap(_returns)(cost)

        #flatten before normalization (per batch) but after discounting (per episode)
        cost = cost.flatten()

        if normalize_cost_returns:
            cost = (cost-cost.mean(-1, keepdims=True))/(cost.std(-1, keepdims=True)+1e-8)


        #no - sign because we assume a cost instead of a reward
        loss = np.sum(jax.lax.stop_gradient(cost)*logp.flatten())


        #L1 penalty on weights
        if lambda_l1 > 0.:
            reg = jax.tree_map(lambda x: np.abs(x).sum(), eqx.filter(model, eqx.is_array))
            reg = lambda_l1 * jax.tree_util.tree_reduce(lambda x,y: x+y, reg)
            loss = loss + reg
        

        return loss

    return _reinforce_loss
    





############################################################################################################
# Works the same as ReinforceLoss in principle, but no easy compatibility with eqx.filter_value_and_grad
# (eqx differentiates wrt to self and no easy way to change that)
############################################################################################################

# import abc


# class _Loss():

#     @abc.abstractmethod
#     def __call__(self, model, istate, *, key, n_sim_steps=1, **kwargs):
#         pass



# class _ReinforceLoss(_Loss):
#     cost_fn:        eqx.field(static=True)
#     batch_size:     eqx.field(static=True)
#     gamma:          eqx.field(static=True)
#     lambda_l1:      eqx.field(static=True)
#     norm_cost_returns: eqx.field(static=True)


#     def __init__(self, cost_fn, *, batch_size=1, gamma=.9, lambda_l1=0., norm_cost_returns=True):

#         self.cost_fn = cost_fn
#         self.batch_size = int(batch_size)
#         self.gamma = float(gamma)
#         self.lambda_l1 = float(lambda_l1)
#         self.norm_cost_returns = bool(norm_cost_returns)


#     def __call__(self, model, istate, *, key, n_sim_steps=1, **kwargs):

#         subkeys = jax.random.split(key, self.batch_size)
#         subkeys = np.asarray(subkeys)

#         vsim = jax.vmap(partial(simulate, history=True), (None, None, 0, None))
#         trajectory, logp = vsim(model, istate, subkeys, n_sim_steps)


#         istate = jtu.tree_map(lambda x: np.repeat(x[None,None,:,:],self.batch_size,0), istate)

#         trajectory = jtu.tree_map(lambda *v: np.concatenate(v,1), *[istate, trajectory])

#         cost = jax.vmap(self.cost_fn)(trajectory)


#         #discounted costs
#         def _returns(costs):
#             def _scan_fn(G, c):
#                 G = G*self.gamma + c
#                 return G, G
#             _, Gs = jax.lax.scan(_scan_fn, np.asarray(0.), costs[::-1])
#             return Gs[::-1]
        
        
#         if self.gamma > 0.:
#             cost = jax.vmap(_returns)(cost)

#         #flatten before normalization (per batch) but after discounting (per episode)
#         cost = cost.flatten()

#         if self.norm_cost_returns:
#             cost = (cost-cost.mean(-1, keepdims=True))/(cost.std(-1, keepdims=True)+1e-8)


#         #no - sign because we assume a cost instead of a reward
#         loss = np.sum(jax.lax.stop_gradient(cost)*logp.flatten())


#         #L1 penalty on weights
#         if self.lambda_l1 > 0.:
#             reg = jax.tree_map(lambda x: np.abs(x).sum(), eqx.filter(model, eqx.is_array))
#             reg = self.lambda_l1 * jax.tree_util.tree_reduce(lambda x,y: x+y, reg)
#             loss = loss + reg
        

#         return loss