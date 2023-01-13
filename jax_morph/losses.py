import jax.numpy as np
from jax import lax

import equinox as eqx

from .state_metrics import *
from .simulation import simulation, sim_trajectory

###----------------------------------------------------------


@eqx.filter_jit
@eqx.filter_vmap(default=None, kwargs=dict(sim_key=0))
def simple_loss(params, 
            hyper_params,
            fstep,
            fspace,
            istate,
            sim_key=None,
            metric_fn=diff_n_ctypes,
            target_metric=0.,
           ):
    '''
    Only calculates the deterministic part of the square loss, does not manage stochastic nodes. Loss given only on last state. 
    '''

    # merge params dicts
    all_params = eqx.combine(params, hyper_params)
    
    #simulation length
    ncells_add = all_params['ncells_add']

    #forward pass - simulation
    sim_init, sim_step = simulation(fstep, all_params, fspace)
    fstate, _ = sim_trajectory(istate, sim_init, sim_step, ncells_add, sim_key)

    # Calculate metric of final structure
    metric_final = metric_fn(fstate)

    # measure difference between final state and target 
    loss = np.sum(np.power(metric_final - target_metric,2))

    return loss




@eqx.filter_jit
@eqx.filter_vmap(default=None, kwargs=dict(sim_key=0))
def reinforce_loss(params, 
            hyper_params,
            fstep,
            fspace,
            istate,
            sim_key=None,
            metric_fn=diff_n_ctypes,
            target_metric=0.,
            GAMMA=.9,
           ):
    '''
    REINFORCE loss with discounting. Loss given only on last state. 
    '''

    # merge params dicts
    all_params = eqx.combine(params, hyper_params)
    
    #simulation length
    ncells_add = all_params['ncells_add']


    #forward pass - simulation
    sim_init, sim_step = simulation(fstep, all_params, fspace)
    fstate, logp = sim_trajectory(istate, sim_init, sim_step, ncells_add, sim_key)

    # Calculate metric of final structure
    metric_final = metric_fn(fstate)

    # measure difference between final state and target 
    loss = np.sum(np.power(metric_final - target_metric,2)) 
    
    #discount losses (loss only given at last timestep)
    steps = len(logp)
    discounted_losses = np.array([(GAMMA**(steps-i))*loss for i in np.arange(steps)])
    
    loss = np.sum(logp*lax.stop_gradient(discounted_losses))

    return loss




@eqx.filter_jit
@eqx.filter_vmap(default=None, kwargs=dict(sim_key=0))
def combined_loss(params, 
            hyper_params,
            fstep,
            fspace,
            istate,
            sim_key=None,
            metric_fn=diff_n_ctypes, 
            target_metric=0.,
            GAMMA=.9,
           ):
    '''
    REINFORCE loss with discounting and loss gradient contribution. Loss given only on last state. 
    '''

    # merge params dicts
    all_params = eqx.combine(params, hyper_params)
    
    #simulation length
    ncells_add = all_params['ncells_add']

    #forward pass - simulation
    sim_init, sim_step = simulation(fstep, all_params, fspace)
    fstate, logp = sim_trajectory(istate, sim_init, sim_step, ncells_add, sim_key)

    # Calculate metric of final structure
    metric_final = metric_fn(fstate)

    # measure difference between final state and target 
    loss = np.sum(np.power(metric_final - target_metric,2))
    
    #discount losses (loss only given at last timestep)
    steps = len(logp)
    discounted_losses = np.array([(GAMMA**(steps-i))*loss for i in np.arange(steps)])
    
    return loss + np.sum(logp*lax.stop_gradient(discounted_losses))





###############################################################
# ENSEMBLING
###############################################################

#@eqx.filter_jit #NO JIT IS FASTER IN THIS CASE
def avg_loss(params, hyper_params, vloss_fn, sim_keys, **kwargs):
    
    lss = vloss_fn(params, hyper_params, sim_key=sim_keys, **kwargs)
    
    return np.mean(lss)