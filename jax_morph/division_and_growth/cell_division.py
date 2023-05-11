import jax
import jax.numpy as np
from jax import random

import jax_md.dataclasses as jdc


def S_no_division(state, params, fspace):
    '''
    Does nothing. Needed for the case where no division happens since simulation still expects a state and log_p.
    '''
    return state, 0.


def S_cell_division(state, params, fspace=None):#, ST_grad=False):
    '''
    Performs one cell division with probability proportional to the current state divrates.
    '''

    def _divide(): 
    
        cellRadBirth = params['cellRadBirth'] #easier to reuse
        
        #split key
        new_key, subkey_div, subkey_place = random.split(state.key,3)
        
        p = state.divrate/state.divrate.sum()
        

        ### DOESN'T WORK SINCE ALL NUMBERS IN GRAD CALCULATION ARE CONVERTED TO FLOAT32
        ### SO OBV USELESS TO INDEX ARRAYS
        # # straight-through estimator set grad of sampling to 1
        # if ST_grad:
        #     def _sample_ST(p, subkey):
        #         #select cells that divides
        #         idx_dividing_cell = random.choice(subkey, a=len(p), p=p)
        #         zero = np.sum(p - jax.lax.stop_gradient(p))
        #         return zero + jax.lax.stop_gradient(idx_dividing_cell)
            
        #     idx_dividing_cell = _sample_ST(p, subkey_div).astype(np.int32)
        # else:
        
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
        
        return new_state, log_p
    
    
    def _no_division():
        return state, 0.
    
    return jax.lax.cond(state.divrate.sum()>0, _divide, _no_division)




def S_cell_div_indep(state, params, fspace=None):#, ST_grad=False):
    '''
    Performs one cell division with probability proportional to the current state divrates.
    '''

    def _divide(args):

        state, idx_dividing_cell = args
    
        cellRadBirth = params['cellRadBirth'] #easier to reuse
        
        idx_new_cell = np.count_nonzero(state.celltype)
        
        ### POSITION OF NEW CELLS
        #note that cell positions will be symmetric so max is pi

        key, subkey_place = random.split(state.key)

        angle = random.uniform(subkey_place, minval=0., maxval=np.pi, dtype=np.float32)

        first_cell = np.array([np.cos(angle),np.sin(angle)])
        second_cell = np.array([-np.cos(angle),-np.sin(angle)])
        
        pos1 = state.position[idx_dividing_cell] + cellRadBirth*first_cell
        pos2 = state.position[idx_dividing_cell] + cellRadBirth*second_cell
        
        
        new_fields = {}
        for field in jdc.fields(state):

            value = getattr(state, field.name)

            if 'position' == field.name:
                new_fields[field.name] = value.at[idx_dividing_cell].set(pos1).at[idx_new_cell].set(pos2)
            elif 'radius' == field.name:
                new_fields[field.name] = value.at[idx_dividing_cell].set(cellRadBirth).at[idx_new_cell].set(cellRadBirth)
            elif 'key' == field.name:
                new_fields[field.name] = key
            else:
                new_fields[field.name] = value.at[idx_new_cell].set(value[idx_dividing_cell])

        new_state = type(state)(**new_fields)
        
        return new_state
    
    
    def _no_division(args):
        state, _ = args
        return state
    

    #split key
    key, subkey_div = random.split(state.key)
    state = jdc.replace(state, key=key)
    
    p = state.divrate

    dividing_cells = random.uniform(subkey_div, (state.celltype.shape)) < state.divrate

    log_p = np.sum(np.log(np.where(dividing_cells, state.divrate, 1-state.divrate)))

    for i in np.arange(state.celltype.shape[0]):
        state = jax.lax.cond(dividing_cells[i], _divide, _no_division, (state, i))

    
    return state, log_p