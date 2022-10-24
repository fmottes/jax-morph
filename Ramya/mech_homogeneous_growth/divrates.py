import jax.numpy as np
from jax import jit, lax, vmap

import jax_md.dataclasses as jax_dataclasses
from jax_md import partition, util, smap, space, energy
from jax_morph.utils import logistic
from jax_morph.datastructures import CellState

maybe_downcast = util.maybe_downcast

def stress(dr,sigma,epsilon,alpha, radius):
  """force arising from Morse interaction between particles with an equilirbium distance at sigma.
  Args:
    dr: distance between two particles.
    sigma: Distance between particles where the energy has a minimum.
    epsilon: Interaction energy scale.
    alpha: Range parameter.
  Returns:
    force between two cells
  """
  angle = np.arctan2(dr[1] - dr[1] + 1e-5, dr[0] - dr[0])
  F = -2* epsilon * alpha * np.exp(-alpha*(dr - sigma))*( np.exp(-alpha*(dr - sigma))- np.float32(1) )
  F_x = F*np.cos(angle)
  F_y = F*np.sin(angle)
  area = np.pi*np.power(radius, 2)
  sigma_xx = F_x*dr[0]/area
  sigma_yy = F_y*dr[1]/area
  sigma_xy = F_x*dr[1]/area
  sigma_yx = F_y*dr[0]/area
  stress_tensor = np.array([[sigma_xx, sigma_xy], [sigma_yx, sigma_yy]])
  # For now, return sum of stress tensor
  return np.nan_to_num(np.sum(stress_tensor, dtype=dr.dtype))

def stress_neighbor_list(
    displacement_or_metric,
    box_size,
    species=None,
    sigma=1.0,
    epsilon=5.0,
    alpha=5.0,
    radius = 0.5,
    r_onset=2.0,
    r_cutoff=2.5,
    dr_threshold=0.5,
    per_particle=False,
    fractional_coordinates=False,
    format: partition.NeighborListFormat=partition.OrderedSparse,
    ):

  """Convenience wrapper to compute morse force using a neighbor list."""
  sigma = maybe_downcast(sigma)
  epsilon = maybe_downcast(epsilon)
  alpha = maybe_downcast(alpha)
  r_onset = maybe_downcast(r_onset)
  r_cutoff = maybe_downcast(r_cutoff)
  radius = maybe_downcast(radius)
  dr_threshold = maybe_downcast(dr_threshold)

  stress_fn = smap.pair_neighbor_list(
    energy.multiplicative_isotropic_cutoff(stress, r_onset, r_cutoff),
    space.canonicalize_displacement_or_metric(displacement_or_metric),
    ignore_unused_parameters=True,
    species=species,
    sigma=sigma,
    epsilon=epsilon,
    alpha=alpha,
    radius=radius,
    reduce_axis=(1,) if per_particle else None)

  return stress_fn

def _generate_morse_params_twotypes(state, params):
    '''
    Morse interaction params for each particle couple. 

    Returns:
      sigma_matrix: Distance between particles where the energy has a minimum.
      epsilon_matrix: Depth of Morse well. 
    '''

    epsilon_OneOne = params['eps_OneOne']
    epsilon_TwoTwo = params['eps_TwoTwo']
    epsilon_OneTwo = params['eps_OneTwo']


    #minimum energy when the two cells barely touch
    #minimum of the well is (approx) at the sum of the two radii (sure??)
    radii = np.array([state.radius]) 
    sigma_matrix = radii+radii.T

    #calculate epsilon (well depth) for each pair based on type
    celltypeOne = np.array([np.where(state.celltype==1,1,0)]) 
    celltypeTwo = np.array([np.where(state.celltype==2,1,0)]) 
    
    epsilon_matrix = np.outer(celltypeOne , celltypeOne)* epsilon_OneOne + \
                   np.outer(celltypeTwo , celltypeTwo)* epsilon_TwoTwo + \
                   np.outer(celltypeOne , celltypeTwo)* epsilon_OneTwo + \
                   np.outer(celltypeTwo, celltypeOne)* epsilon_OneTwo 

    return epsilon_matrix, sigma_matrix

def div_mechanical(state,fspace, params, nbrs) -> np.array:
    div_gamma = params['div_gamma']
    div_k = params['div_k']
    
    # Calculate stresses
    epsilon_matrix, sigma_matrix = _generate_morse_params_twotypes(state, params)
    # TODO: do i need to specify box size? 
    stress_fn = stress_neighbor_list(fspace.displacement,  fspace.box_size, 
    sigma=sigma_matrix, epsilon=epsilon_matrix, alpha=params['alpha'], radius=state.radius, 
    r_onset=params['r_onset'], r_cutoff=params['r_cutoff'])
    stresses = stress_fn(state.position, nbrs)
    # calculate "rates"
    div = logistic(stresses,div_gamma[0],div_k[0])
    div = np.where(stresses > 0, div, logistic(stresses,div_gamma[1],div_k[1]))
    
    # create array with new divrates
    divrate = np.where(state.celltype>0,divrate, 0.0)
    
    # cells cannot divide if they are too small
    # constants are arbitrary, change if you change cell radius
    divrate = divrate*logistic(state.radius+.06, 50, params['cellRad'])
    
    return divrate