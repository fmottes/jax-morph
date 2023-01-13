import jax.numpy as np
from jax import vmap, random, value_and_grad, lax
import optax
import equinox as eqx
from jax_morph.simulation import simulation, sim_trajectory

''' Coefficient of variation of division rates loss.'''
def cv_divrates(state):
    return np.std(state.divrate)/np.mean(state.divrate)

'''Run simulation with given parameters and calculate loss of final state.'''
@eqx.filter_jit
@eqx.filter_vmap(default=None, kwargs=dict(sim_key=0))
def simple_loss(params, 
            hyper_params,
            fstep,
            fspace,
            istate,
            sim_key=None,
            metric_fn=cv_divrates,
            target_metric=0.,
           ):

    # Merge params dicts.
    all_params = eqx.combine(params, hyper_params)

    # Forward pass - run simulation.
    sim_init, sim_step = simulation(fstep, all_params, fspace)
    fstate, _ = sim_trajectory(istate, sim_init, sim_step, sim_key)

    # Calculate metric of final structure.
    metric_final = metric_fn(fstate)

    # Measure difference between final state and target. 
    loss = np.sum(np.power(metric_final - target_metric,2))

    return loss

@eqx.filter_jit
@eqx.filter_vmap(default=None, kwargs=dict(sim_key=0))
def combined_loss(params, 
            hyper_params,
            fstep,
            fspace,
            istate,
            sim_key=None,
            metric_fn=cv_divrates,#diff_avg_divrates, 
            target_metric=0.,
            GAMMA=.9,
           ):
    '''
    REINFORCE loss with discounting and loss gradient contribution. 
    '''

    # merge params dicts
    all_params = eqx.combine(params, hyper_params)

    #forward pass - simulation
    sim_init, sim_step = simulation(fstep, all_params, fspace)
    fstate, logp = sim_trajectory(istate, sim_init, sim_step, sim_key)

    # Calculate metric of final structure
    metric_final = metric_fn(fstate)

    # measure difference between final state and target 
    loss = np.sum(np.power(metric_final - target_metric,2))
    
    #discount losses (loss only given at last timestep)
    steps = len(logp)
    discounted_losses = np.array([(GAMMA**(steps-i))*loss for i in np.arange(steps)])
    return loss + np.sum(logp*lax.stop_gradient(discounted_losses))

'''Average loss over a batch of simulations with different seeds.'''
@eqx.filter_jit #NO JIT IS FASTER IN THIS CASE
def avg_loss(params, hyper_params, vloss_fn, sim_keys, **kwargs):
    
    lss = vloss_fn(params, hyper_params, sim_key=sim_keys, **kwargs)
    
    return np.mean(lss)

'''Optimization loop.'''
def optimize(key, epochs, batch_size, lr, params, train_params, fstep, fspace, istate, **kwargs):
    # Separate params to be optimized.
    p, hp = eqx.partition(params, train_params)

    # Set up optimizer.
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr)) 
    opt_state = optimizer.init(p)

    # Generate batch keys.
    key, *batch_subkeys = random.split(key, batch_size+1)
    batch_subkeys = np.array(batch_subkeys)

    # JIT'ted grad function
    vg_jit = eqx.filter_jit(value_and_grad(avg_loss))
    # Get starting gradients and loss.
    ll, grads = vg_jit(p, hp, combined_loss, batch_subkeys, fstep=fstep, fspace=fspace, istate=istate)
    l = avg_loss(p, hp, simple_loss, batch_subkeys, fstep=fstep, fspace=fspace, istate=istate)
    #l, grads = value_and_grad(avg_loss)(p, hp, simple_loss, batch_subkeys, fstep=fstep, fspace=fspace, istate=istate)
    print("loss: %s, reinforce: %s" % (l, ll))
    params_t = [p]
    loss_t = [float(l)]
    grads_t = [grads]

    # Optimization loop.
    for t in range(epochs):
        
        key, *batch_subkeys = random.split(key, batch_size+1)
        batch_subkeys = np.array(batch_subkeys)
        updates, opt_state = optimizer.update(grads, opt_state, p)
        p = eqx.apply_updates(p, updates)
        ll, grads = vg_jit(p, hp, combined_loss, batch_subkeys, fstep=fstep, fspace=fspace, istate=istate)
        l = avg_loss(p, hp, simple_loss, batch_subkeys, fstep=fstep, fspace=fspace, istate=istate)
        #l, grads = value_and_grad(avg_loss)(p, hp, simple_loss, batch_subkeys, fstep=fstep, fspace=fspace, istate=istate)
        print("loss: %s, reinforce: %s" % (l, ll))
        
        # Store values for each epoch.
        loss_t.append(float(l))
        params_t.append(p)
        grads_t.append(grads)

    return params_t, loss_t, grads_t