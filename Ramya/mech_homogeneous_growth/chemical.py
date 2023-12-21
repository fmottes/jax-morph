import jax.numpy as np
import jax_md.dataclasses as jax_dataclasses
from jax_morph.datastructures import CellState
from jax_morph.utils import logistic
from jax import lax, vmap, random
from jax_md import util, quantity
from traitlets.config.loader import FileConfigLoader
f32 = util.f32


def S_fixed_chemfield(state,
                  params,
                  fspace,
                  **kwargs 
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
  center = np.array([0.0, 0.0])
  chemfield_disp = vmap(fspace.displacement, (0, None))(state.position, center)
  chemfield_disp = np.linalg.norm(chemfield_disp, axis=1)
  chemfield = 50.0/(2 + 0.4*np.power(chemfield_disp, 2))
  chemfield = np.where(state.celltype > 0, chemfield, 0.0) 
  state = jax_dataclasses.replace(state, field=chemfield)
  return state
