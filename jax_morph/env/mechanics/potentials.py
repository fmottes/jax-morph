import jax
import jax.numpy as np

import jax_md

import equinox as eqx


import abc
from typing import Union

from functools import partial



#Define Potential ABC
class MechanicalInteractionPotential(eqx.Module):

    @abc.abstractmethod
    def energy_fn(self, state, *, per_particle):
        pass




from functools import partial
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
        # Morse potential with neighbor list
        epsilon_matrix = self._calculate_epsilon_matrix(state)
        sigma_matrix = self._calculate_sigma_matrix(state)

        #generate morse pair potential
        _, morse_energy = jax_md.energy.morse_neighbor_list(state.displacement,
                                                box_size=10.0,
                                                alpha=self.alpha,
                                                epsilon=epsilon_matrix,
                                                sigma=sigma_matrix, 
                                                r_onset=self.r_onset, 
                                                r_cutoff=self.r_cutoff,
                                                per_particle=per_particle
                                                )       
        return morse_energy

class MorsePotentialSpecies(MechanicalInteractionPotential):
    epsilon:   Union[float, jax.Array] = 3.
    alpha:     float = 2.8
    r_cutoff:  float = eqx.field(default=2., static=True)
    r_onset:   float = eqx.field(default=1.7, static=True)


    def _calculate_epsilon_matrix(self, state):

        if np.atleast_1d(self.epsilon).size == 1:
            alive = np.where(state.celltype.sum(1) > 0, 1, 0)
            epsilon_matrix = (np.outer(alive, alive)-np.eye(alive.shape[0]))*self.epsilon
        else:
            # Parametrize epsilon matrix so it's symmetric - leave the diagonals alone and take average of symmetric off diagonal elements
            epsilon_matrix = .5*(np.triu(self.epsilon) + np.tril(self.epsilon).T + np.triu(self.epsilon).T + np.tril(self.epsilon))
            epsilon_matrix = epsilon_matrix - np.eye(state.celltype.shape[1])*.5*np.diagonal(epsilon_matrix)
            epsilon_matrix = jax.nn.sigmoid(epsilon_matrix)*10. + .8
            #if (state.celltype.shape[-1] + 1) != self.epsilon.shape[0]:
            #    raise AssertionError('Epsilon matrix does not include species for non-alive cells')
            #epsilon_matrix = self.epsilon
            #epsilon_matrix = epsilon_matrix.at[0, :].set(0.0).at[:,0].set(0.0)
            # Constrain value of epsilon within reasonable range

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
        # Morse potential with neighbor list
        epsilon_matrix = self._calculate_epsilon_matrix(state)
        sigma_matrix = self._calculate_sigma_matrix(state)

        species = np.argmax(state.celltype, axis=1).astype(int)
        species = np.where(state.celltype.sum(-1) > 0.0, species, 0)
        
        #generate morse pair potential
        _, morse_energy = jax_md.energy.morse_neighbor_list(state.displacement,
                                                box_size=10.0,
                                                alpha=self.alpha,
                                                epsilon=epsilon_matrix,
                                                sigma=sigma_matrix, 
                                                r_onset=self.r_onset, 
                                                r_cutoff=self.r_cutoff,
                                                per_particle=per_particle,
                                                species=species, 
                                                )       
        return morse_energy


class MorsePotentialCadherin(MechanicalInteractionPotential):
    #epsilon:   Union[float, jax.Array] = 2.
    alpha:     float = 2.8
    r_cutoff:  float = eqx.field(default=2., static=True)
    r_onset:   float = eqx.field(default=1.7, static=True)


    def _calculate_epsilon_matrix(self, state):
        num_ctypes = state.celltype.shape[1]
        epsilon_matrix = np.reshape(state.cadherin, (num_ctypes, num_ctypes))
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
        # Morse potential with neighbor list
        epsilon_matrix = self._calculate_epsilon_matrix(state)
        sigma_matrix = self._calculate_sigma_matrix(state)

        species = np.argmax(state.celltype, axis=1).astype(int)
        species = np.where(state.celltype.sum(-1) > 0.0, species, 0)
        
        #generate morse pair potential
        _, morse_energy = jax_md.energy.morse_neighbor_list(state.displacement,
                                                species=species,
                                                box_size=10.0,
                                                alpha=self.alpha,
                                                epsilon=epsilon_matrix,
                                                sigma=sigma_matrix, 
                                                r_onset=self.r_onset, 
                                                r_cutoff=self.r_cutoff,
                                                per_particle=per_particle
                                                )   
        return morse_energy