import jax.numpy as np
import jax_md.dataclasses as jax_dataclasses
from jax_morph.datastructures import CellState
from jax_morph.utils import logistic
from jax import lax, vmap
from jax_md import util, quantity
f32 = util.f32


def S_fixed_chemfield(istate,
                  params,
                  fspace 
                  ) -> CellState:
  """
  Fixed morphogen field based on particle position from center cell.
  Arg:
    state: current state
    param: dictionary with parameters
  Returns:
    big_state_output: CellState with updated chemical concentration
  """
  # Find displacements from center of cluster.
  cluster_box_size = quantity.box_size_at_number_density(params['ncells_init'], 1.2, 2)
  center = np.array([cluster_box_size/2.0, cluster_box_size/2.0])
  chemfield_disp = vmap(fspace.displacement, (0, None))(istate.position, center)
  chemfield_disp = np.linalg.norm(chemfield_disp, axis=1)
  # TODO: Write these out as params
  chemfield_conc = 100.0/(2.0 + 0.4*np.power(chemfield_disp, 2.0))
  chemfield_conc = np.reshape(np.where(istate.celltype > 0, chemfield_conc, 0.0), (-1, 1))
  istate = jax_dataclasses.replace(istate, chemical=chemfield_conc)
  return istate