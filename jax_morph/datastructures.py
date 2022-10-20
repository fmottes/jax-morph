import jax_md.dataclasses as jax_dataclasses
from jax_md import util, space
Array = util.Array



@jax_dataclasses.dataclass
class CellState:
    '''
    Dataclass containing the system state.

    Attributes:
    
    position: The current position of the particles. An ndarray of floats with
              shape [n, spatial_dimension].
    celltype: The cell type of each particle. An ndarray of integers in [0,1] with 
              shape [n, 1]
    radius:    Radius of each particle. Cells are born at cellRadBirth and grow up to cellRad
    chemical: Chemical concentration at location of each particle. An ndarray of 
              np.float32 with shape [n, numSigs] integers in [0,1] with shape [n, 1]
    key:      The current state of the random number generator.
    '''
    
    position: Array
    celltype: Array
    radius: Array 
    chemical: Array
    divrate: Array
    key: Array
    
@jax_dataclasses.dataclass
class SpaceFunc:
    '''
    Dataclass containing functions that handle space.
    '''
    
    displacement: space.DisplacementFn
    shift: space.ShiftFn