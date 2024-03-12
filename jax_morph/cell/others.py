import jax
import jax.numpy as np

import equinox as eqx

from .._base import SimulationStep



class SecretionMaskByCellType(SimulationStep):
    ctype_sec_chem:     eqx.field(static=True)

    def return_logprob(self) -> bool:
        return False

    def __init__(self, state, ctype_sec_chem=None):

        if ctype_sec_chem is None:
            self.ctype_sec_chem = np.repeat(np.atleast_2d([1.]*state.chemical.shape[1]), state.celltype.shape[-1], axis=0).tolist()

        else:

            if np.asarray(ctype_sec_chem).shape != (state.celltype.shape[1], state.chemical.shape[1]):
                raise ValueError("ctype_sec_chem must be shape (N_CELLTYPE, N_CHEM)")
            
            self.ctype_sec_chem = ctype_sec_chem
        
        
    @jax.named_scope("jax_morph.SecretionByCellType")
    def __call__(self, state, *, key=None, **kwargs):
            
        sec_mask = state.celltype @ np.atleast_2d(self.ctype_sec_chem)

        secretion_rate = sec_mask*state.secretion_rate

        state = eqx.tree_at(lambda s: s.secretion_rate, state, secretion_rate)

        return state