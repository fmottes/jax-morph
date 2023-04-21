import jax
import jax.numpy as np

import jax_md
import jax_md.dataclasses as jdc

from jax_morph.division_and_growth.cell_division import S_cell_division
from jax_morph.division_and_growth.cell_growth import S_grow_cells
from jax_morph.mechanics.morse import S_mech_morse_relax




def init_state_grow(key, empty_state, params, fspace, n_cells=5):
    '''
    Initialize an empty state with a single cell and grow to a given number of cells.

    NOTE: empty_state must include the following fields for this initialization method to work correctly:
    - position
    - celltype
    - radius
    - divrate

    All other fields are initialized to zero. All cells are set to the same radius and celltype.
    '''

    assert n_cells > 0, 'Must initialize at least one cell.'

    # elongate datastructures to the accomodate the initial number of cells
    new_fields = {}
    for field in jdc.fields(empty_state):

        if field.name == 'key':
            new_fields[field.name] = key

        else:
            #retrieve the value of the field
            value = getattr(empty_state, field.name)

            if jax_md.util.is_array(value):

                if len(value.shape) > 0:
                    shape = (n_cells,)+(value.shape[1:])
                    new_fields[field.name] = np.zeros(shape, dtype=value.dtype)
                    
                else:
                    new_fields[field.name] = value
            else:
                new_fields[field.name] = value


    state = type(empty_state)(**new_fields)


    # initialize the first cell
    celltype = state.celltype.at[0].set(1)
    radius = state.radius.at[0].set(params['cellRad'])
    divrate = state.divrate.at[0].set(1.)

    state = jdc.replace(state, celltype=celltype, radius=radius, divrate=divrate)

    
    # add one cell at a time and relax the system
    def _init_add(state, i):
        state, _    = S_cell_division(state, params, fspace)
        state       = S_grow_cells(state, params, fspace)
        state       = S_mech_morse_relax(state, params, fspace)
        return state, 0.
    
    iterations = np.arange(n_cells-1)
    state, _ = jax.lax.scan(_init_add, state, iterations)
    
    
    #set all cells to max radius and relax the system
    radius = np.ones_like(state.radius)*params['cellRad']
    state = jdc.replace(state, radius=radius)
    
    state = S_mech_morse_relax(state, params, fspace)

    #set key to None to signal possibly inconsistent state
    state = jdc.replace(state, key=None)
    
    return state





# def _create_onecell_state(key, params):
#### NOT FLEXIBLE ENOUGH
#### SET DEFAULTS IN CELLSTATE DEFINITION INSTEAD