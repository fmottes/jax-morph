import jax
import jax.numpy as np

import equinox as eqx

from ._base import SimulationStep

from typing import Callable, Union, Sequence




###------------SEQUENTIAL SIMULATION STEP-----------------###


class Sequential(SimulationStep):
    substeps: tuple
    _return_logp: bool = eqx.field(static=True)
    _return_nbrs: bool = eqx.field(static=True)

    def return_logprob(self) -> bool:
        return self._return_logp

    def return_nbrs(self) -> bool:
        return self._return_nbrs

    def __init__(self, substeps: Sequence[Callable]):

        if all(isinstance(x, SimulationStep) for x in substeps):
            self.substeps = tuple(substeps)
        else:
            raise TypeError("All substeps must be of type `SimulationStep`")
        
        self._return_logp = any(x.return_logprob() for x in self.substeps)
        self._return_nbrs = any(x.return_nbrs() for x in self.substeps)        


    @jax.named_scope("jax_morph.Sequential")
    def __call__(self, state, *, nbrs=None, key=None, **kwargs):

        if key is None:
            keys = [None] * len(self.substeps)
        else:
            keys = jax.random.split(key, len(self.substeps))


        logp = np.float_(0.)
        for substep, key in zip(self.substeps, keys):

            #TODO: NEED TO FIX THIS CONTROL FLOW
            
            if substep.return_logprob():
                state, logp = substep(state, key=key)
                logp += logp
            elif substep.return_nbrs():
                state, nbrs = substep(state, nbrs, key=key)
            else:
                state = substep(state, key=key)


        if self._return_logp and self._return_nbrs:
            return state, logp, nbrs
        elif (not self._return_logp) and self._return_nbrs:
            return state, nbrs
        elif self._return_logp and (not self._return_nbrs):
            return state, logp
        else:
            return state
        



    def __getitem__(self, i: Union[int, slice]) -> Callable:
        if isinstance(i, int):
            return self.substeps[i]
        elif isinstance(i, slice):
            return Sequential(self.substeps[i])
        else:
            raise TypeError(f"Indexing with type {type(i)} is not supported")

    def __iter__(self):
        yield from self.substeps

    def __len__(self):
        return len(self.substeps)
    


###------------SIMULATION FUNCTION-----------------###

@eqx.filter_jit
def simulate(model, state, key, n_steps=1, *, nbrs=None, history=False):
    subkeys = jax.random.split(key, n_steps)
    
    # Currently assuming that everything is using neighbor list
    if model.return_nbrs():
        
        #STOCHASTIC MODEL
        if model.return_logprob():
            def _scan_fn(sstate, k):
                state, nbrs = sstate
                state, logp, nbrs = model(state, nbrs=nbrs, key=k)
                sstate = (state, nbrs)
                return sstate, (state, logp, nbrs)
            
            (state, nbrs), (trajectory, logp, nbrs) = jax.lax.scan(_scan_fn, (state, nbrs), np.asarray(subkeys))
    
            if history:
                return trajectory, logp
            else:
                return state, logp
                
        #DETERMINISTIC (OR REPARAMETRIZED) MODEL
        else:
            def _scan_fn(sstate, k):
                state, nbrs = sstate
                state, nbrs = model(state, nbrs=nbrs, key=k)
                sstate = (state, nbrs)
                return sstate, (state, nbrs)
            
            (state, nbrs), trajectory = jax.lax.scan(_scan_fn, (state, nbrs), np.asarray(subkeys))
    
            if history:
                return trajectory
            else:
                return state

    else:
        raise TypeError("Must use neighbor list!")