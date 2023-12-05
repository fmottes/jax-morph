import jax
import jax.numpy as np

import jax_md

import equinox as eqx

from jax_morph.simulation import SimulationStep

import abc
from typing import Union





#Define Potential ABC
class MechanicalInteractionPotential(eqx.Module):

    @abc.abstractmethod
    def energy_fn(self, state, *, per_particle):
        pass




class MorsePotential(MechanicalInteractionPotential):
    epsilon:   Union[float, jax.Array] = 3.
    alpha:     float = 2.8
    r_cutoff:  float = eqx.field(default=2., static=True)
    r_onset:   float = eqx.field(default=1.7, static=True)


    def _calculate_epsilon_matrix(self, state):

        if np.atleast_1d(self.epsilon).size == 1:
            alive = np.where(state.celltype.sum(1) > 0, 1, 0)
            epsilon_matrix = (np.outer(alive, alive)-np.eye(alive.shape[0]))*self.epsilon


        elif isinstance(self.epsilon, jax.interpreters.xla.DeviceArray):
            
            ### implement logic for multiple cell types
            raise NotImplementedError('Multiple cell types not implemented yet')


        return epsilon_matrix
    

    def _calculate_sigma_matrix(self, state):

        # sigma_matrix = state.radius[:,None] + state.radius[None,:]

        # In principle this should be the right expression:
        # alive = np.where(state.celltype.sum(1) > 0, 1, 0)
        # sigma_matrix = (state.radius + state.radius.T - 2*np.diag(test_state.radius.squeeze()))*np.outer(alive, alive)
        
        # BUT since we already put epsilon = 0 for non-interacting cells, we can save on some operations

        sigma_matrix = state.radius + state.radius.T

        return sigma_matrix
    

    def energy_fn(self, state, *, per_particle=False):

        epsilon_matrix = self._calculate_epsilon_matrix(state)
        sigma_matrix = self._calculate_sigma_matrix(state)

        #generate morse pair potential
        morse_energy = jax_md.energy.morse_pair(state.displacement,
                                                alpha=self.alpha,
                                                epsilon=epsilon_matrix,
                                                sigma=sigma_matrix, 
                                                r_onset=self.r_onset, 
                                                r_cutoff=self.r_cutoff,
                                                per_particle=per_particle
                                                )
        
        return morse_energy
