import jax
import jax.numpy as np

import equinox as eqx

from ..simulation import simulation, sim_trajectory


##########################################################################


@eqx.filter_jit
@eqx.filter_vmap(default=None, kwargs=dict(sim_key=0))
def loss(params, 
         hyper_params,
         fstep,
         fspace,
         istate,
         sim_key=None,
         metric_fn=None,
         metric_type='reward',
         REINFORCE=True,
         GAMMA=.99,
         ncells_add=None
         ):
    '''
    Reinforce loss on trajectory (with discounting). Rewards are differences in successive state metrics.

    If REINFORCE=False, then the loss is just the state measure on the final state.

    GAMMA is the discount factor for the calculation of the returns.

    If metric_type='reward', it is maximized, if metric_type='cost', it is minimized.

    '''

    #simulation length
    ncells_add = hyper_params['ncells_add'] if ncells_add is None else ncells_add
    
    def _sim_trajectory(istate, sim_init, sim_step, ncells_add, key=None):

        state = sim_init(istate, ncells_add, key)

        def scan_fn(state, i):
            state, logp = sim_step(state)
            measure = metric_fn(state)
            return state, (logp, measure)


        iterations = np.arange(ncells_add)
        fstate, aux = jax.lax.scan(scan_fn, state, iterations)

        return fstate, aux

    # merge params dicts
    all_params = eqx.combine(params, hyper_params)

    #forward pass - simulation
    sim_init, sim_step = simulation(fstep, all_params, fspace)
    _, (logp, measures) = _sim_trajectory(istate, sim_init, sim_step, ncells_add, sim_key)

    
    if REINFORCE:
        
        def _returns_rec(rewards):
            Gs=[]
            G=0
            for r in rewards[::-1]:
                G = r+G*GAMMA
                Gs.append(G)

            return np.array(Gs)[::-1]
        
        
        measures = np.append(np.array([metric_fn(istate)]),measures)
        
        if metric_type=='reward':
            rewards = np.diff(measures)
        elif metric_type=='cost':
            rewards = -np.diff(measures)


        returns = _returns_rec(rewards)

        # standardizing returns helps with convergence
        returns = (returns-returns.mean())/(returns.std()+1e-8)

        loss = -np.sum(logp*jax.lax.stop_gradient(returns))

        return loss

    else:
        return measures[-1]





##########################################################################
# (REINFORCE LOSS ON TRAJECTORY | L2 LOSS ON FINAL STATE) + REGULARIZATION
##########################################################################

@eqx.filter_jit
@eqx.filter_vmap(default=None, kwargs=dict(sim_key=0))
def reinforce_loss(params, 
                   hyper_params,
                   fstep,fspace,
                   istate,
                   sim_key=None,
                   metric_fn=None,
                   target_metric=0.,
                   REINFORCE=True,
                   LAMBDA=0., #no regularization by default
                   GAMMA=.95,
                   ncells_add=None
                   ):
    '''
    Reinforce loss on trajectory (with discounting). Rewards based on l2 loss on metric_fn.

    If REINFORCE=False, then the loss is just the l2 loss on the final state.

    If LAMBDA>0, then the loss is regularized with L2 norm of the weights of the division and secretion NNs.

    GAMMA is the discount factor for the calculation of the returns.

    '''

    #simulation length
    ncells_add = hyper_params['ncells_add'] if ncells_add is None else ncells_add
    
    def _sim_trajectory(istate, sim_init, sim_step, ncells_add, key=None):

        state = sim_init(istate, ncells_add, key)

        def scan_fn(state, i):
            state, logp = sim_step(state)
            loss = np.sum((metric_fn(state) - target_metric)**2)
            loss = np.sqrt(loss)
            return state, (logp, loss)


        iterations = len(state.celltype)-len(istate.celltype)
        iterations = np.arange(iterations)
        fstate, aux = jax.lax.scan(scan_fn, state, iterations)

        return fstate, aux

    # merge params dicts
    all_params = eqx.combine(params, hyper_params)

    #forward pass - simulation
    sim_init, sim_step = simulation(fstep, all_params, fspace)
    fstate, (logp, losses) = _sim_trajectory(istate, sim_init, sim_step, ncells_add, sim_key)

    
    if REINFORCE:
        
        def _returns_rec(losses):
            Gs=[]
            G=0
            for l in losses[::-1]:
                G = l+G*GAMMA
                Gs.append(G)

            return np.array(Gs)[::-1]
        
        iloss = np.sqrt(np.sum((metric_fn(istate) - target_metric)**2))
        rewards = -np.diff(np.append(np.array([iloss]),losses))
        
        #discount rewards
        returns = _returns_rec(rewards)
        
        loss = -np.sum(logp*jax.lax.stop_gradient(returns))

        # possibly L2-regularize NN weights
        if LAMBDA > 0.:
            if 'div_fn' in params:
                loss += LAMBDA*np.array(jax.tree_leaves(jax.tree_map(lambda x: np.sum(x**2), params['div_fn']))).sum()
            if 'sec_fn' in params:
                loss += LAMBDA*np.array(jax.tree_leaves(jax.tree_map(lambda x: np.sum(x**2), params['sec_fn']))).sum()

    else:
        loss = losses[-1]

    return loss




###############################################################
# ENSEMBLING
###############################################################

#@eqx.filter_jit #NO JIT IS FASTER IN THIS CASE
def avg_loss(params, hyper_params, vloss_fn, sim_keys, **kwargs):
    
    lss = vloss_fn(params, hyper_params, sim_key=sim_keys, **kwargs)
    
    return np.mean(lss)





###############################################################
# BASIC L2 LOSS
###############################################################

@eqx.filter_jit
@eqx.filter_vmap(default=None, kwargs=dict(sim_key=0))
def l2_loss(params, 
            hyper_params,
            fstep,
            fspace,
            istate,
            sim_key=None,
            metric_fn=None,
            target_metric=0.,
            ncells_add=None,
           ):
    '''
    DEPRECATED.

    Only calculates the deterministic part of the square loss, does not manage stochastic nodes. 
    Loss given only on last state. 
    '''

    # merge params dicts
    all_params = eqx.combine(params, hyper_params)
    
    #simulation length
    ncells_add = hyper_params['ncells_add'] if ncells_add is None else ncells_add

    #forward pass - simulation
    sim_init, sim_step = simulation(fstep, all_params, fspace)
    fstate, _ = sim_trajectory(istate, sim_init, sim_step, ncells_add, sim_key)

    # Calculate metric of final structure
    metric_final = metric_fn(fstate)

    # measure difference between final state and target 
    loss = np.sum(np.power(metric_final - target_metric,2))

    return loss




###############################################################
# DEPRECATED
# REINFORCE LOSS ONLY ON FINAL STATE
###############################################################

# @eqx.filter_jit
# @eqx.filter_vmap(default=None, kwargs=dict(sim_key=0))
# def reinforce_loss(params, 
#             hyper_params,
#             fstep,
#             fspace,
#             istate,
#             sim_key=None,
#             metric_fn=diff_n_ctypes,
#             target_metric=0.,
#             GAMMA=.9,
#            ):
#     '''
#     REINFORCE loss with discounting. Loss given only on last state. 
#     '''

#     # merge params dicts
#     all_params = eqx.combine(params, hyper_params)
    
#     #simulation length
#     ncells_add = all_params['ncells_add']


#     #forward pass - simulation
#     sim_init, sim_step = simulation(fstep, all_params, fspace)
#     fstate, logp = sim_trajectory(istate, sim_init, sim_step, ncells_add, sim_key)

#     # Calculate metric of final structure
#     metric_final = metric_fn(fstate)

#     # measure difference between final state and target 
#     loss = np.sum(np.power(metric_final - target_metric,2)) 
    
#     #discount losses (loss only given at last timestep)
#     steps = len(logp)
#     discounted_losses = np.array([(GAMMA**(steps-i))*loss for i in np.arange(steps)])
    
#     loss = np.sum(logp*jax.lax.stop_gradient(discounted_losses))

#     return loss




#################################################################
# DEPRECATED
# REINFORCE LOSS ONLY ON FINAL STATE + LOSS GRADIENT CONTRIBUTION
# (useless if loss is not differentiable anyway)
#################################################################

# @eqx.filter_jit
# @eqx.filter_vmap(default=None, kwargs=dict(sim_key=0))
# def combined_loss(params, 
#             hyper_params,
#             fstep,
#             fspace,
#             istate,
#             sim_key=None,
#             metric_fn=diff_n_ctypes, 
#             target_metric=0.,
#             GAMMA=.9,
#            ):
#     '''
#     REINFORCE loss with discounting and loss gradient contribution. Loss given only on last state. 
#     '''

#     # merge params dicts
#     all_params = eqx.combine(params, hyper_params)
    
#     #simulation length
#     ncells_add = all_params['ncells_add']

#     #forward pass - simulation
#     sim_init, sim_step = simulation(fstep, all_params, fspace)
#     fstate, logp = sim_trajectory(istate, sim_init, sim_step, ncells_add, sim_key)

#     # Calculate metric of final structure
#     metric_final = metric_fn(fstate)

#     # measure difference between final state and target 
#     loss = np.sum(np.power(metric_final - target_metric,2))
    
#     #discount losses (loss only given at last timestep)
#     steps = len(logp)
#     discounted_losses = np.array([(GAMMA**(steps-i))*loss for i in np.arange(steps)])
    
#     return loss + np.sum(logp*jax.lax.stop_gradient(discounted_losses))
