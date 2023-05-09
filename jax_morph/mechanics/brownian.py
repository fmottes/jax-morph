import jax
import jax.numpy as np
from jax import jit, lax

from jax_md import simulate
import jax_md.dataclasses as jdc



def S_mech_brownian(state, params, fspace, build_energy=None, dt=1e-3, n_steps=None, kT=1e-3, gamma=.3):

    energy_fn = build_energy(state, params, fspace) #needed since energy_fn changes after cell divisions

    n_steps = n_steps if n_steps is not None else params['mech_relaxation_steps'] #1000

    init, apply = simulate.brownian(energy_fn, fspace.shift, dt, kT, gamma)

    key, subkey = jax.random.split(state.key)

    br_state = init(subkey, state.position)

    def _step(i, _state):
        _state = apply(_state)
        return _state
    
    br_state = lax.fori_loop(0, n_steps, _step, br_state)

    pos = np.where(state.celltype==0, state.position, br_state.position)

    state = jdc.replace(state, position=pos, key=key)

    return state