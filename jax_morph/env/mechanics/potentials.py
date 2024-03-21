import jax
import jax.numpy as np

import jax_md

import equinox as eqx


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


class MorsePotentialSpecies(MechanicalInteractionPotential):
    """Morse potential function using celltype as species"""
    epsilon:   Union[float, jax.Array] = 3.
    alpha:     float = 2.8
    r_cutoff:  float = eqx.field(default=2., static=True)
    r_onset:   float = eqx.field(default=1.7, static=True)


    def _calculate_epsilon_matrix(self, state):

        if np.atleast_1d(self.epsilon).size == 1:
            alive = np.where(state.celltype.sum(1) > 0, 1, 0)
            epsilon_matrix = (np.outer(alive, alive)-np.eye(alive.shape[0]))*self.epsilon
        else:
            if self.epsilon.shape[0] != state.celltype.shape[1] or self.epsilon.shape[1] != state.celltype.shape[1]:
                raise ValueError("Epsilon matrix is not n_ctype x n_ctype to use species morse potential function.")
            
            # First parametrize epsilon matrix so it's symmetric - leave the diagonals alone and take average of symmetric off diagonal elements
            epsilon_matrix = self.epsilon
            epsilon_matrix = .5*(np.triu(self.epsilon) + np.tril(self.epsilon).T + np.triu(self.epsilon).T + np.tril(self.epsilon))
            epsilon_matrix = epsilon_matrix - np.eye(state.celltype.shape[1])*.5*np.diagonal(epsilon_matrix)
            epsilon_matrix = jax.nn.sigmoid(epsilon_matrix)*10. + .8

            # Now turn this into N x N array of pairwise epsilons
            epsilon_matrix = state.celltype @ epsilon_matrix
            epsilon_matrix = epsilon_matrix @ state.celltype.T
            epsilon_matrix = np.where(state.celltype.sum(-1) > 0., epsilon_matrix, 0.0)

        return epsilon_matrix
    

    def _calculate_sigma_matrix(self, state):
        
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
                                                per_particle=per_particle,
                                                )       
        return morse_energy


class MorsePotentialChemical(MechanicalInteractionPotential):
    """Use cadherin values to calculate morse potential"""
    alpha:     float = 2.8
    r_cutoff:  float = eqx.field(default=2., static=True)
    r_onset:   float = eqx.field(default=1.7, static=True)

    def symmetrize_matrix(self, matrix):
        def _symmetrize_matrix(matrix):
            matrix = .5*(np.triu(matrix) + np.tril(matrix).T + np.triu(matrix).T + np.tril(matrix))
            matrix = matrix - np.eye(matrix.shape[0])*.5*np.diagonal(matrix)
            return matrix
        return jax.vmap(_symmetrize_matrix)(matrix)
    
    def _calculate_epsilon_matrix(self, state):

        epsilon_matrix = np.reshape(state.epsilon, (state.celltype.shape[0], 
                                                    state.celltype.shape[1], 
                                                    state.celltype.shape[1]))
        
        epsilon_matrix =  self.symmetrize_matrix(epsilon_matrix)
        
        def _get_epsilon(ctype_i, ctype_j, eps_i, eps_j):
            eps_one = ctype_i[None, :] @ eps_i.squeeze() @ ctype_j[:,None]
            eps_two = ctype_j[None, :] @ eps_j.squeeze() @ ctype_i[:,None]
            return .5*(eps_one.squeeze() + eps_two.squeeze())

        epsilon_matrix = jax.vmap(jax.vmap(_get_epsilon, (0,None, 0, None)), (None,0, None, 0))(state.celltype, state.celltype, epsilon_matrix, epsilon_matrix)
        epsilon_matrix = 5.*jax.nn.sigmoid(epsilon_matrix) + .3
        epsilon_matrix = np.where(state.celltype.sum(-1) > 0., epsilon_matrix, 0.0) 
        
        return epsilon_matrix
    

    def _calculate_sigma_matrix(self, state):
        sigma_matrix = state.radius + state.radius.T

        return sigma_matrix

    def energy_fn(self, state, *, per_particle=False):
        
        # Morse potential with neighbor list
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