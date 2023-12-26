import jax
import jax.numpy as np

import jax_md

import equinox as eqx

from .._base import SimulationStep

from typing import Union





###-----------CHEMICAL GRADIENTS-----------------###

class LocalChemicalGradients(SimulationStep):
    neighbor_radius:    Union[float,None] = eqx.field(static=True)


    def return_logprob(self) -> bool:
        return False


    def __init__(self, *, neighbor_radius=None, **kwargs):

        self.neighbor_radius = neighbor_radius


    @jax.named_scope("jax_morph.LocalChemicalGradients")
    def __call__(self, state, *, key=None, **kwargs):

        # mask only cells that exist
        c_alive = state.celltype.sum(1)>0.
        c_alive = np.outer(c_alive, c_alive)

        # displacements between cell pairs (ncells x ncells x ndim_space)
        disp = jax.vmap(jax.vmap(state.displacement, in_axes=[0,None]), in_axes=[None,0])(state.position, state.position)

        # distances btw cell pairs
        dist = (disp**2).sum(2)

        # avoid division by zero in gradient calculation
        safe_dist = np.where(dist>0., dist, 1)
        dist = np.where(dist>0., safe_dist, 0)

        # dist w/ non-existing cells are zeroed out
        dist *= c_alive

        #jax.debug.print('{x}', x=(disp**2).sum(2))

        # consider as neigbors:
        # - cells less than one radius away (+ small tolerance)
        # - cells differents from themselves
        # - cells that exist
        if None == self.neighbor_radius:
            # "touching" distance betw. cells
            R = (state.radius+state.radius.T)*c_alive
        else:
            R = (self.neighbor_radius)*c_alive
        
        neig = (dist<R)*(dist>0.)

        #safe_dist = np.where(dist>0, dist, 1)

        # normalize all displacements
        norm_disp = (disp*neig[:,:,None])/(dist[:,:,None]+1e-8)
        norm_disp = np.where((dist>0)[:,:,None], (disp*neig[:,:,None])/safe_dist[:,:,None], 0)

        # calculates x and y components of grad for single chemical
        def _grad_chem(chem):
            return (norm_disp*chem.ravel()[:,None]).sum(1)
            

        #vectorize over chemicals
        #OUTPUT SHAPE: ncells x ndim x nchem
        _grad_chem = jax.vmap(_grad_chem, in_axes=1, out_axes=2)

        #calc grads (no non-existing cells or lone cells w/ no neighbors)
        chemgrads = _grad_chem(state.chemical)
            
        # transform into ncells x (grad_x + grad_y)
        #reshape like ncells x ndim x nchem to revert
        chemgrads = chemgrads.reshape(state.celltype.shape[0], -1)

        #update state
        state = eqx.tree_at(lambda s: s.chemical_grad, state, chemgrads)

        return state
    



###-----------MECHANICAL "STRESS"-----------------###

class LocalMechanicalStress(SimulationStep):
    mechanical_potential:       eqx.Module


    def return_logprob(self) -> bool:
        return False


    def __init__(self, mechanical_potential, **kwargs):
        self.mechanical_potential = mechanical_potential

    
    @jax.named_scope("jax_morph.LocalMechanicalStress")
    def __call__(self, state, *, key=None, **kwargs):

        #generate pair potential
        pair_potential = self.mechanical_potential.energy_fn(state)
        
        forces = jax.jacrev(pair_potential)(state.position)
        
        # F_ij = force on i by j, r_ij = displacement from i to j
        drs = jax_md.space.map_product(state.displacement)(state.position, state.position)
        
        stresses = np.sum(np.multiply(forces, np.sign(drs)), axis=(0, 2))[:,None] / 1e2 #heuristic rescaling for compatibility with other cell inputs
        stresses = np.where(state.celltype.sum(1)[:,None] > 0, stresses, 0.)

        state = eqx.tree_at(lambda s: s.mechanical_stress, state, stresses)

        return state