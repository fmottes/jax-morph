import jax.numpy as np
import jax_md.dataclasses as jax_dataclasses
from jax_morph.datastructures import CellState
from jax_morph.utils import logistic
from jax import lax, vmap, random
from jax_md import util, quantity
from traitlets.config.loader import FileConfigLoader
f32 = util.f32


def S_fixed_chemfield(istate,
                  params,
                  fspace,
                  noise=0.0,
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
  chemfield_disp = vmap(fspace.displacement, (0, None))(istate.position, center)
  chemfield_disp = np.linalg.norm(chemfield_disp, axis=1)
  # TODO: Write these out as params
  chemfield = params["chem_max"]/(params["chem_k"] + params["chem_gamma"]*np.power(chemfield_disp, 2.0))
  chemfield = np.where(istate.celltype > 0, chemfield, 0.0) + noise*random.normal(istate.key, chemfield.shape)
  istate = jax_dataclasses.replace(istate, field=chemfield)
  return istate