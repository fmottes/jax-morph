from jax_morph.datastructures import SpaceFunc
from Francesco.chem_twotypes.initial_states import init_state_grow 
from jax_morph.simulation import sim_trajectory
import jax.numpy as np
from jax import random, jacrev, jit, vmap, lax, tree_map
from optax import adam, apply_updates
from jax_md import quantity, space

def target_cv(quantities):
   return np.std(quantities, axis=1)/np.mean(quantities, axis=1)
 
def optimize(params, key,
                 target_value, grad_descent_steps,
                 target_fn, sim_init,
                 sim_step, step_size,
                 n_ensemble,
                 ):
 
   # Set up optimizer
   opt = adam(step_size)
   opt_state = opt.init(params)
 
   # initialize array to store loss & param values in subsequent optimization steps
   lossAll = np.zeros(grad_descent_steps)
   paramsAll = []
 
   # get key and split it into # in ensemble keys
   nkeys = n_ensemble*grad_descent_steps
   key_array  = np.reshape(random.split(key,nkeys),
   (grad_descent_steps, n_ensemble, -1))
   
   # size for the number of initial cells
   box_size = quantity.box_size_at_number_density(params['ncells_init'] + params['ncells_add'], 1.2, 2)
   init_box_size = quantity.box_size_at_number_density(params['ncells_init'], 1.5, 2)
 
   # build initial state and space handling functions
   fspace = SpaceFunc(*space.periodic(box_size))
   fspace_init = SpaceFunc(*space.periodic(init_box_size))

   def loss(key, params, target_value, target_fn, sim_init, sim_step, fspace, fspace_init):
            # create initial state
        istate = init_state_grow(key, params, fspace_init)
            # run entire simulation
        traj, log_p = sim_trajectory(istate, sim_init, sim_step)
        target_loss = np.average(np.power((target_cv(traj.divrate) - target_value), 2))
        return target_loss, log_p
   def get_gradient(key, params, target_value, target_fn, sim_init, sim_step, fspace, fspace_init):
   # define loss as average loss over simulation ensemble
        target_loss, log_p = loss(key, params, target_value, target_fn, sim_init, sim_step, fspace, fspace_init)
        reinforce_loss = tree_map(lambda x, y: x + y,
            tree_map(lambda x: x*lax.stop_gradient(target_loss), log_p),
            target_loss)
        return reinforce_loss

   grad_reinforce = jacrev(get_gradient, 1)(key, params, target_value, target_fn, sim_init, sim_step, fspace, fspace_init)
   get_gradient_vmapped = vmap(get_gradient, (0, None, None, None, None, None, None, None))
   def do_opt_step(step,params, opt_state):
     results = get_gradient_vmapped(key, params, target_value, target_fn, sim_init, sim_step, fspace, fspace_init)
     grad_reinforce = tree_map(lambda x: np.average(x), results)
     #flat_x, unravel = ravel_pytree(opt_state)
     ls = loss(key, params, target_value, target_fn, sim_init, sim_step, fspace, fspace_init)
     print('\nstep: ', step, ', ls: ', ls, '\ng_loss:', grad_reinforce)
     updates, opt_state = opt.update(grad_reinforce, opt_state)
     params = apply_updates(params, updates)
     return ls, params, opt_state
 
   for step in range(grad_descent_steps):
     ls, params, opt_state = do_opt_step(step, params, opt_state)
     lossAll = lossAll.at[step].set(ls)
     paramsAll.append(params)
   return params, lossAll, paramsAll
