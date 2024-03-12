from jax import random, vmap
from jax.random import split
import jax.numpy as np
from jax_md import space, quantity, util
import jax_md.dataclasses as jdc
from jax.nn import sigmoid
########## IMPORT JAX-MORPH FUNCTIONS ##########
################################################

from jax_morph.datastructures import SpaceFunc
from jax_morph.utils import _maybe_array, logistic
from jax_morph.simulation import simulation, sim_trajectory

# IMPORT STATE-CHANGING FUNCTIONS
from jax_morph.division_and_growth.cell_division import S_cell_division, S_cell_div_indep, S_cell_div_indep_MC
from jax_morph.division_and_growth.cell_growth import S_grow_cells

from jax_morph.mechanics.morse import S_mech_morse_relax
from jax_morph.cell_internals.stress import S_set_stress
from jax_morph.chemicals.secdiff import S_ss_chemfield

from jax_morph.cell_internals.divrates import S_set_divrate, div_nn
from jax_morph.cell_internals.secretion import sec_nn
from jax_morph.cell_internals.grad_estimate import S_chemical_gradients
from jax_morph.cell_internals.hidden_state import hidden_state_nn, S_hidden_state
from jax_morph.chemicals.diffusion import diffuse_allchem_ss_exp
from jax_morph.initial_states import init_state_grow

from jax_morph.visualization import draw_circles_ctype, draw_circles_chem, draw_circles_divrate, draw_circles
from Ramya.mech_homogeneous_growth.chemical import S_fixed_chemfield

from functools import partial
from collections import namedtuple
import equinox as eqx
import haiku as hk



def default_params(key, n_chem=2):

    ### CELL DIMENSIONS
    cellRad = .5
    cellRadBirth = float(cellRad / np.sqrt(2))


    ### DIFFUSION

    # No diffusion or secretion in my simulation - only external chemical field over positions
    diffCoeff = np.ones(n_chem) 
    degRate = np.ones(n_chem) 

    # diffusion cutoff
    r_cutoffDiff = 5.*cellRad
    r_onsetDiff = r_cutoffDiff - .5

    # CHEMICAL FIELD
    chem_max = 50.0

    ### SECRETION

    # sec rate that gives concentration 1 at source at SS
    #sec_max_unitary = 2*np.sqrt(diffCoeff*degRate)

    sec_max = np.ones((n_chem,), dtype=np.float32)
    #sec_max = sec_max.at[0].set(10) 
    #secreted_by_ctypes = np.ones((n_chem, 1))
    ctype_sec_chem = np.ones((1, 2))

    # GROWTH


    # MORSE POTENTIAL
    # always use python scalars
    alpha = 3.0
    eps_OneOne = 3.


    # morse cutoff
    r_cutoff = 5.*cellRad
    r_onset = r_cutoff - .2


    # number of gradient descent steps for Morse potential minimization
    mech_relaxation_steps = 10


    # Initialization and number of added cells. 
    ncells_init = 100 #number of cells in the initial cluster
    ncells_add = 100

    hidden_state_size = 8
    hid_hidden = [8,]
    div_hidden = []
    sec_hidden = []
    hidden_state_decay = 0.0

    #@title Define trainable params
    train_params = {
        'n_chem': False,
        'n_dim': False,
        'sec_max': True,
        'ctype_sec_chem': False,

        
        'cellRad' : False,
        'cellRadBirth' : False,
        
        'diffCoeff' : True,
        'degRate' : False,
        'r_onsetDiff' : False,
        'r_cutoffDiff' : False,
        
        'alpha': False, 
        'eps_OneOne': False,
        'r_onset' : False,
        'r_cutoff' : False,
        'mech_relaxation_steps' : False,
        
        'ncells_init' : False,
        'ncells_add': False,

        'chem_max': False, 
        'hidden_state_size': False,
    }

    #@title Initialize params
    params = {
        'n_chem': n_chem,
        'n_dim': 2,
        'sec_max': sec_max,
        'ctype_sec_chem' : ctype_sec_chem,
        
        'cellRad' : cellRad,
        'cellRadBirth' : cellRadBirth,
        
        'diffCoeff' : diffCoeff,
        'degRate' : degRate,
        'r_onsetDiff' : r_onsetDiff,
        'r_cutoffDiff' : r_cutoffDiff,
        
        'alpha': _maybe_array('alpha', alpha, train_params), 
        'eps_OneOne': _maybe_array('eps_OneOne', eps_OneOne, train_params),
        'r_onset' : r_onset,
        'r_cutoff' : r_cutoff,
        'mech_relaxation_steps' : mech_relaxation_steps,
        
        'ncells_init' : ncells_init,
        'ncells_add': ncells_add,

        'chem_max': chem_max,

        'hidden_state_size':  hidden_state_size,

    }

    return params, train_params

# decorator MUST be jax_md.dataclass instead of dataclasses.dataclass
# to make dataclass compatible with jax tree operations
@jdc.dataclass
class CellState:
    '''
    Dataclass containing the system state.

    STATE
    -----

    '''

    # STATE
    position:   util.Array
    celltype:   util.Array
    radius:     util.Array
    chemical:   util.Array
    chemgrad:   util.Array
    field:     util.Array
    stress:   util.Array
    hidden_state: util.Array
    divrate:    util.Array
    key:        util.Array


    @classmethod
    def default_init(cls, n_dim=2, n_chem=1, hidden_size=10):
        '''
        Intializes a CellState with no cells (empty data structures, with correct shapes).
        

        Parameters
        ----------
        n_dim: int
            Number of spatial dimensions.
        n_chem: int
            Number of chemical species.

        Returns
        -------
        CellState
        '''

        assert n_dim == 2 or n_dim == 3, 'n_dim must be 2 or 3'
        assert n_chem > 0 and isinstance(n_chem, int), 'n_chem must be a positive integer'
        
        defaultstate = {
            'position'  :   np.empty(shape=(0, n_dim),              dtype=np.float32),
            'celltype'  :   np.empty(shape=(0,),                    dtype=np.int8),
            'radius'    :   np.empty(shape=(0,),                    dtype=np.float32),
            'chemical'  :   np.empty(shape=(0, n_chem),             dtype=np.float32),
            'chemgrad'  :   np.empty(shape=(0, int(n_dim*n_chem)),  dtype=np.float32),
            'field'   :      np.empty(shape=(0,),                   dtype=np.float32),
            'stress'  :   np.empty(shape=(0,),                      dtype=np.float32), 
            'hidden_state' : np.empty(shape=(0, hidden_size),       dtype=np.float32),
            'divrate'   :   np.empty(shape=(0,),                    dtype=np.float32),
            'key'       :   None,
        }


        return cls(**defaultstate)

def build_sim_from_params(params, train_params, key):

    N_CELLS_INIT = params['ncells_init']
    HID_HIDDEN = [8,]
    DIV_HIDDEN = []
    SEC_HIDDEN = []
    HID_STATE_DECAY = 0.0


    ## Initialize CellState
    fspace = SpaceFunc(*space.free())
    istate = CellState.default_init(n_dim=params['n_dim'], 
                                    n_chem=params['n_chem'],
                                    hidden_size=params['hidden_state_size']
                                    )
    key, init_key = split(key)
    istate = init_state_grow(init_key, istate, params, fspace, N_CELLS_INIT)


    ### ANN: Hidden state
    use_state_fields = CellState(position=      False, 
                                celltype=      False, 
                                radius=            False, 
                                chemical=          True,
                                chemgrad=          True,
                                field=             False,
                                stress=            True,
                                divrate=           False,
                                hidden_state=  False,
                                key=           False
                                )


    hid_init, hid_nn_apply = hidden_state_nn(params,
                                            train_params,
                                            HID_HIDDEN,
                                            use_state_fields,
                                            train=True,
                                            transform_mlp_out=None,
                                            )

    key, init_key = split(key)
    params, train_params = hid_init(istate, init_key)


    ### ANN: Division
    use_state_fields = CellState(position=      False, 
                                celltype=      False, 
                                radius=        False, 
                                chemical=      False,
                                chemgrad=      False,
                                field=         False,
                                stress=        False,
                                divrate=       False,
                                hidden_state=      True,
                                key=           False
                                )

    transform_fwd = lambda state, divrates: divrates*logistic(state.field, 0.1, 25.0)
    div_init, div_nn_apply = div_nn(params,
                                    train_params,
                                    DIV_HIDDEN,
                                    use_state_fields,
                                    train=True,
                                    w_init=hk.initializers.Constant(0),
                                    transform_mlp_out=sigmoid,
                                    transform_fwd=transform_fwd,
                                    )

    key, init_key = split(key)
    params, train_params = div_init(istate, init_key)


    ### ANN: Secretion
    use_state_fields = CellState(position=      False, 
                                celltype=      False, 
                                radius=        False, 
                                chemical=      False,
                                chemgrad=      False,
                                field=         False,
                                stress=        False,
                                divrate=       False,
                                hidden_state=      True,
                                key=           False
                                )

    sec_init, sec_nn_apply = sec_nn(params,
                                    train_params,
                                    SEC_HIDDEN,
                                    use_state_fields,
                                    train=True,
                                    w_init=hk.initializers.Orthogonal(),
                                    #w_init=hk.initializers.Constant(0)
                                    )

    key, init_key = split(key)
    params, train_params = sec_init(istate, init_key)



    ### Simulation loop
    fstep = [
        # ENV CHANGES
        S_cell_division,
        S_grow_cells,
        partial(S_mech_morse_relax, dt=.0001),
        partial(S_ss_chemfield, sec_fn=sec_nn_apply, diffusion_fn=diffuse_allchem_ss_exp,n_iter=3),

        # SENSING
        #chemicals sensed directly
        S_chemical_gradients,
        S_fixed_chemfield,
        S_set_stress,
        # INTERNAL (HIDDEN) STATE
        partial(S_hidden_state, dhidden_fn=eqx.filter_jit(hid_nn_apply), state_decay=.0),
        # POLICIES
        partial(S_set_divrate, divrate_fn=eqx.filter_jit(div_nn_apply))
    ]


    #sim_init, sim_step = simulation(fstep, params, fspace)

    SimulBlocks = namedtuple('SimulBlocks', ['fstep', 'fspace', 'istate', 'params', 'train_params'])

    return SimulBlocks(fstep, fspace, istate, params, train_params) #, sim_init, sim_step
