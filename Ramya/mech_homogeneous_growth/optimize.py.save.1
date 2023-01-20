import jax.numpy as np
from jax import vmap, random, value_and_grad
import optax
import equinox as eqx
from jax_morph.simulation import simulation, sim_trajectory

def cv_divrates(state):
    return np.std(state.divrate)/np.mean(state.divrate)

# does not currently ensemble over different initial states
def _simple_loss_vk(params, 
            hyper_params,
            fstep,
            fspace,
            istate,
            metric_fn=cv_divrates, 
            target_metric = 0.,
            **kwargs
           ):
    
        
        @eqx.filter_jit
        def _simple_loss(sim_key=None):
            '''
            Only calculates the deterministic part of the square loss, does not manage stochastic nodes.
            '''
            # merge params dicts
            all_params = eqx.combine(params, hyper_params)

            #forward pass - simulation
            sim_init, sim_step = simulation(fstep, all_params, fspace, **kwargs)
            fstate, _ = sim_trajectory(istate, sim_init, sim_step, sim_key)

            # Calculate metric of final structure
            metric_final = metric_fn(fstate)

            # measure difference between final state and target 
            loss = np.sum(np.power(metric_final - target_metric,2))

            return loss
        
        return vmap(_simple_loss)

def avg_simple_loss(params, 
            hyper_params,
            fstep,
            fspace,
            istate,
            sim_keys=None,
            metric_fn=cv_divrates,
            target_metric = 0.,
            **kwargs
           ):
    
    loss_fn = _simple_loss_vk(params, 
            hyper_params,
            fstep,
            fspace,
            istate,
            metric_fn,
            target_metric,
            **kwargs
           )
    
    
    losses_ens = loss_fn(sim_key=sim_keys)
    
    return np.mean(losses_ens)

def optimize(key, epochs, batch_size, lr, params, train_params, fstep, fspace, istate, **kwargs):

    p, hp = eqx.partition(params, train_params)
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr)) 
    opt_state = optimizer.init(p)

    #store initial params
    params_t = [p]
    grads_t = []


    #store loss at initial params
    key, *batch_subkeys = random.split(key, batch_size+1)
    batch_subkeys = np.array(batch_subkeys)


    l, grads = value_and_grad(avg_simple_loss)(p, hp, fstep, fspace, istate, batch_subkeys, **kwargs)
    print("loss: %s" % l)
    loss_t = [].astype(float)]



    for t in range(epochs):
        
        key, *batch_subkeys = random.split(key, batch_size+1)
        batch_subkeys = np.array(batch_subkeys)
        updates, opt_state = optimizer.update(grads, opt_state, p)
        p = eqx.apply_updates(p, updates)
        
        l, grads = value_and_grad(avg_simple_loss)(p, hp, fstep, fspace, istate, batch_subkeys, **kwargs)    
        print("loss: %s" % l)
        loss_t.append(l.astype(float))
        params_t.append(p)
        grads_t.append(grads)

    return params_t, loss_t, grads_t
