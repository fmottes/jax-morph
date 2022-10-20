import jax.numpy as np
from jax import jit, lax, random

import jax_md.dataclasses as jax_dataclasses

from jax_morph.datastructures import CellState
from jax_morph.cell_division import S_cell_division
from jax_morph.cell_growth import S_grow_cells

from jax_md import minimize, energy, quantity

# Set initial_attribute of existing cells to all ones. 
def _set_initial_attribute(n_dim, params):
  """
  Returns array of ones that cen be used as initial attribute for CellState - 
  all initialized cells set to 1 and non-exising cells set to zero. 
  Arg: 
    n_dim: Number of dimensions of the attribute
  Returns:
    attr: nCells x n_dim array
  """
  N = params['ncells_init'] + params['ncells_add']
  ncells_init = params['ncells_init']
  attr = np.ones((ncells_init, n_dim), np.float32)
  attr = np.vstack((attr, np.zeros((N - ncells_init, n_dim))))
  attr = np.squeeze(attr)
  return attr

def _relax_initial_config(R, fspace):
  """
  Uses fire descent and soft sphere energy to relax initial structure 
  configuration.
  Arg:
    R: initial positions
  Returns:
    R: relaxed positions
  """
  energy_fn = energy.soft_sphere_pair(fspace.displacement)
  init_fn, step_fn = minimize.fire_descent(energy_fn, fspace.shift)
  state = init_fn(R)
  state = lax.while_loop(
      lambda state: np.max(np.abs(state.force)) > 1e-1,
      step_fn,
      state
  )
  R = np.squeeze(R)
  return R

def _template_initial_state(key, 
                           field_ndims, 
                           init_positions_fn,
                           fspace,
                           params):
  """
  Template initial state to start with. Sets up random positions and relaxes configuration
  and sets CellState attributes to ones. 
  Arg:
    key: jax.random.PRNGKey 
    field_ndims: array specifying dimensions of each attribute
  Returns:
    CellState
  """
  R = init_positions_fn(key, fspace, params)
  attributes = [R,]
  for ndim in field_ndims: 
    attributes.append(_set_initial_attribute(ndim, params))
  # Random key.
  attributes.append(key)
  return CellState(*attributes)

def init_packed_positions(key, fspace, params):
  N = params['ncells_init'] + params['ncells_add']
  cluster_box_size = quantity.box_size_at_number_density(params['ncells_init'], 1.2, 2)
  R = random.uniform(key, (N, 2))*cluster_box_size
  R = _relax_initial_config(R, fspace)
  return R

def packed_cell_state(key, params, fspace):
  """
    Creates packed cell state - all nCellsInit are closely packed together
    Arg:
      key: jax.random.PRNGKey
      getGrowth_fn: function to assign initial growthrates with
    Returns:
      big_state
  """

  field_ndims = np.ones
  # TODO: this in a more intelligent way
  field_ndims = np.array([1, 1, params['n_chem'], 1, 1]).astype(np.int32)
  big_state = _template_initial_state(key, field_ndims, init_packed_positions, fspace, params)

  # Set only positions of existing cells. 
  big_state = jax_dataclasses.replace(big_state,
                                  position = np.where(np.reshape(big_state.celltype > 0, (-1, 1)), big_state.position, np.array([[0.0, 0.0]]))
  )
  N, ncells_init, cellRad, n_chem = params['ncells_init'] + params['ncells_add'], params['ncells_init'], params['cellRad'], params['n_chem']

  # TODO: Also set growthrates.
  # Set all radii. 
  big_state = jax_dataclasses.replace(big_state,
                                  radius=big_state.radius*cellRad)
  # TODO: Need to update chemical concentrations here if there's a field.
  # Chemical has to be 2D for some reason. 
  big_state = jax_dataclasses.replace(big_state, 
                                  chemical=np.reshape(big_state.chemical, (-1, 1)))
  big_state = jax_dataclasses.replace(big_state, 
                                  division=0.0*big_state.division)
  return big_state

