import jax_md
import jax_md.dataclasses as jdc



@jdc.dataclass
class CellState:
    '''
    Example dataclass containing the system state.

    Attributes:
    
    position: The current position of the particles. An ndarray of floats with
              shape [n, spatial_dimension].
    celltype: The cell type of each particle. An ndarray of integers in [0,1] with 
              shape [n, 1]
    radius:    Radius of each particle. Cells are born at cellRadBirth and grow up to cellRad
    chemical: Chemical concentration at location of each particle. An ndarray of 
              np.float32 with shape [n, numSigs] integers in [0,1] with shape [n, 1]
    field:    Imposed external field (like a chemical field)
    key:      The current state of the random number generator.
    '''
    
    position:   jax_md.util.Array
    celltype:   jax_md.util.Array
    radius:     jax_md.util.Array 
    chemical:   jax_md.util.Array
    field:      jax_md.util.Array
    divrate:    jax_md.util.Array
    stress:     jax_md.util.Array
    key:        jax_md.util.Array
    
    
    
@jdc.dataclass
class SpaceFunc:
    '''
    Dataclass containing functions that handle space.
    '''
    
    displacement:   jax_md.space.DisplacementFn
    shift:          jax_md.space.ShiftFn
