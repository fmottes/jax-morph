import jax
import jax.numpy as np
from jax import random

import jax_md.dataclasses as jdc



def S_cell_division(state, params, fspace=None):
    '''
    Performs one cell division with probability proportional to the current state divrates.
    '''

    def _divide(): 
    
        cellRadBirth = params['cellRadBirth'] #easier to reuse
        
        #split key
        new_key, subkey_div, subkey_place = random.split(state.key,3)
        
        p = state.divrate/state.divrate.sum()
        
        #select cells that divides
        idx_dividing_cell = random.choice(subkey_div, a=len(p), p=p)
        
        #save logp for optimization purposes
        log_p = np.log(p[idx_dividing_cell])
        
        idx_new_cell = np.count_nonzero(state.celltype)
        
        ### POSITION OF NEW CELLS
        #note that cell positions will be symmetric so max is pi
        angle = random.uniform(subkey_place, minval=0., maxval=np.pi, dtype=np.float32)

        first_cell = np.array([np.cos(angle),np.sin(angle)])
        second_cell = np.array([-np.cos(angle),-np.sin(angle)])
        
        pos1 = state.position[idx_dividing_cell] + cellRadBirth*first_cell
        pos2 = state.position[idx_dividing_cell] + cellRadBirth*second_cell
        
        # new_position = state.position.at[idx_dividing_cell].set(pos1)
        # new_position = new_position.at[idx_new_cell].set(pos2)
        
        # ### NEW RADII
        # new_radius = state.radius.at[idx_dividing_cell].set(cellRadBirth)
        # new_radius = new_radius.at[idx_new_cell].set(cellRadBirth)
        
        new_fields = {}
        for field in jdc.fields(state):

            value = getattr(state, field.name)

            if 'position' == field.name:
                new_fields[field.name] = value.at[idx_dividing_cell].set(pos1).at[idx_new_cell].set(pos2)
            elif 'radius' == field.name:
                new_fields[field.name] = value.at[idx_dividing_cell].set(cellRadBirth).at[idx_new_cell].set(cellRadBirth)
            elif 'key' == field.name:
                new_fields[field.name] = new_key
            else:
                new_fields[field.name] = value.at[idx_new_cell].set(value[idx_dividing_cell])

        new_state = type(state)(**new_fields)


        # ### INHERIT CELLTYPE
        # new_celltype = state.celltype.at[idx_new_cell].set(state.celltype[idx_dividing_cell])
        
        # # INHERIT CHEMICAL AND DIVRATES
        # # useless, recalculated right after, but just in case
        # new_chemical = state.chemical.at[idx_new_cell].set(state.chemical[idx_dividing_cell])  
        # new_divrate = state.divrate.at[idx_new_cell].set(state.divrate[idx_dividing_cell]) 
        
        # #build new state of the system
        # new_state = jdc.replace(state, 
        #                                     position=new_position,
        #                                     radius=new_radius,
        #                                     celltype=new_celltype,
        #                                     chemical=new_chemical,
        #                                     divrate=new_divrate,
        #                                     key=new_key
        #                                 )
        
        return new_state, log_p
    
    
    def _no_division():
        return state, 0.
    
    return jax.lax.cond(state.divrate.sum()>0, _divide, _no_division)