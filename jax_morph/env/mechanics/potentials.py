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
    alpha:     Union[float, jax.Array] = 2.8
    epsilon_max:   float = 5.0
    epsilon_min:   float = 1.0
    alpha_max:     float = 3.0
    alpha_min:     float = 1.0
    r_cutoff:  float = eqx.field(default=2., static=True)
    r_onset:   float = eqx.field(default=1.7, static=True)



    def _calculate_pairwise_matrix(self, state, matrix, max, min):
        
        if self.epsilon.shape[0] != state.celltype.shape[1] or self.epsilon.shape[1] != state.celltype.shape[1]:
            raise ValueError("Matrix is not n_ctype x n_ctype to use species morse potential function.")
            
        # First parametrize matrix so it's symmetric - take average of symmetric off diagonal elements
        matrix = .5*(np.triu(matrix) + np.tril(matrix).T + np.triu(matrix).T + np.tril(matrix))
        matrix = matrix - np.eye(state.celltype.shape[1])*.5*np.diagonal(matrix)
        matrix = jax.nn.sigmoid(matrix)*max + min

        # Now turn this into N x N array of pairwise
        matrix = state.celltype @ matrix
        matrix = matrix @ state.celltype.T
        matrix = np.where(state.celltype.sum(-1) > 0., matrix, 0.0)

        return matrix
    

    def _calculate_sigma_matrix(self, state):
        
        sigma_matrix = state.radius + state.radius.T

        return sigma_matrix

    def energy_fn(self, state, *, per_particle=False):

        epsilon_matrix = self._calculate_pairwise_matrix(state, self.epsilon, self.epsilon_max, self.epsilon_min)
        alpha_matrix = self._calculate_pairwise_matrix(state, self.alpha, self.alpha_max, self.alpha_min)
        sigma_matrix = self._calculate_sigma_matrix(state)
                        
        #generate morse pair potential
        morse_energy = jax_md.energy.morse_pair(state.displacement,
                                                alpha=alpha_matrix,
                                                epsilon=epsilon_matrix,
                                                sigma=sigma_matrix, 
                                                r_onset=self.r_onset, 
                                                r_cutoff=self.r_cutoff,
                                                per_particle=per_particle,
                                                )       
        return morse_energy


class MorsePotentialCadherin(MechanicalInteractionPotential):
    """Use cadherin values to calculate morse potential"""
    eps_min:   float = eqx.field(default=1., static=True)
    eps_max:   float = eqx.field(default=5., static=True)
    alpha:   float = eqx.field(default=2.8, static=True)
    r_cutoff:  float = eqx.field(default=2., static=True)
    r_onset:   float = eqx.field(default=1.7, static=True)


    def _calculate_epsilon_matrix(self, state):

        ctype_mat = state.celltype@state.celltype.T
        cad_mat = state.celltype*state.cadherin[:,:-1]
        
        # Homotypic interactions
        epsilon_matrix = np.sum(jax.vmap(jax.vmap(lambda x,y: x + y, (0, None)), (None, 0))(cad_mat, cad_mat), axis=-1)
        epsilon_matrix = epsilon_matrix * ctype_mat
        
        # Heterotypic interactions
        epsilon_matrix += (1. - ctype_mat)*(state.cadherin[:,-1] + state.cadherin[:,-1].T)
        
        # Normalize
        epsilon_matrix = self.eps_max*epsilon_matrix + self.eps_min
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