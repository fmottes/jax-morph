import jax
import jax.numpy as np

import equinox as eqx

from ._base import SimulationStep

from typing import Callable, Union, Sequence, Any


###------------SEQUENTIAL SIMULATION STEP-----------------###


class Sequential(SimulationStep):
    substeps: tuple
    _return_logp: bool = eqx.field(static=True)
    _named_substeps: dict = eqx.field(static=True, repr=False)

    def return_logprob(self) -> bool:
        return self._return_logp

    def __init__(self, substeps: Sequence[SimulationStep]):

        super().__init__()

        if not all(isinstance(x, SimulationStep) for x in substeps):
            raise TypeError("All substeps must be of type `SimulationStep`")

        self.substeps = tuple(substeps)
        self._return_logp = any(x.return_logprob() for x in self.substeps)

        # Create a dictionary of named substeps
        self._named_substeps = {}
        for substep in substeps:
            name = substep.__class__.__name__
            if name in self._named_substeps:
                i = 1
                while f"{name}_{i}" in self._named_substeps:
                    i += 1
                name = f"{name}_{i}"
            self._named_substeps[name] = substep

    @jax.named_scope("jax_morph.Sequential")
    def __call__(self, state, *, key=None, **kwargs):

        if key is None:
            keys = [None] * len(self.substeps)
        else:
            keys = jax.random.split(key, len(self.substeps))

        logp = np.float_(0.0)

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

    def __getattr__(self, name: str) -> Any:
        if name in self._named_substeps:
            return self._named_substeps[name]
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

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
def simulate(model, state, key, n_steps=1, *, history=False):

    subkeys = jax.random.split(key, n_steps)

    # STOCHASTIC MODEL
    if model.return_logprob():

        def _scan_fn(state, k):
            state, logp = model(state, key=k)
            return state, (state, logp)

        state, (trajectory, logp) = jax.lax.scan(_scan_fn, state, np.asarray(subkeys))

        if history:
            return trajectory, logp
        else:
            return state, logp

    # DETERMINISTIC (OR REPARAMETRIZED) MODEL
    else:

        def _scan_fn(state, k):
            state = model(state, key=k)
            return state, state

        state, trajectory = jax.lax.scan(_scan_fn, state, np.asarray(subkeys))

        if history:
            return trajectory
        else:
            return state
