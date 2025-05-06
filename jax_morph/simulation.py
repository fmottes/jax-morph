import jax
import jax.numpy as np

import equinox as eqx

from ._base import SimulationStep

from typing import Callable, Union, Sequence




###------------SEQUENTIAL SIMULATION STEP-----------------###


class Sequential(SimulationStep):
    substeps: tuple
    _return_logp: bool = eqx.field(static=True)


    def return_logprob(self) -> bool:
        return self._return_logp



    def __init__(self, substeps: Sequence[Callable]):

        if all(isinstance(x, SimulationStep) for x in substeps):
            self.substeps = tuple(substeps)
        else:
            raise TypeError("All substeps must be of type `SimulationStep`")
        
        self._return_logp = any(x.return_logprob() for x in self.substeps)
        


    @jax.named_scope("jax_morph.Sequential")
    def __call__(self, state, *, key=None, **kwargs):


        if key is None:
            keys = [None] * len(self.substeps)
        else:
            keys = jax.random.split(key, len(self.substeps))


        logp = np.float_(0.)

        for substep, key in zip(self.substeps, keys):

            if substep.return_logprob():
                state, logp = substep(state, key=key)
                logp += logp
            else:
                state = substep(state, key=key)


        if self._return_logp:
            return state, logp
        else:
            return state
        

    def copy(self):
        return Sequential(self.substeps)


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
def simulate(model, state, key, n_steps=1, *, history=False, checkpoint=False):

    subkeys = jax.random.split(key, n_steps)

    #STOCHASTIC MODEL
    if model.return_logprob():

        def _scan_fn(state, k):
            state, logp = model(state, key=k)
            return state, (state, logp)
        
        state, (trajectory, logp) = jax.lax.scan(_scan_fn, state, np.asarray(subkeys))

        if history:
            return trajectory, logp
        else:
            return state, logp
        
    #DETERMINISTIC (OR REPARAMETRIZED) MODEL
    else:
        def _scan_fn(state, k):
            state = model(state, key=k)
            return state, state

        if checkpoint:
            _scan_fn = jax.checkpoint(_scan_fn)
        state, trajectory = jax.lax.scan(_scan_fn, state, np.asarray(subkeys))

        if history:
            return trajectory
        else:
            return state