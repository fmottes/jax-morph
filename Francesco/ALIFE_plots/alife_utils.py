import jax
import jax.numpy as np
from jax import value_and_grad
from jax.random import split

import optax
import equinox as eqx
import haiku as hk

import jax_md
import jax_md.dataclasses as jdc
from jax_md import space

from tqdm import trange
from functools import partial
from collections import namedtuple


### JAX-MORPH IMPORTS ###

from jax_morph.optimization.losses import avg_loss, loss

from jax_morph.datastructures import SpaceFunc
from jax_morph.utils import _maybe_array

from jax_morph.initial_states import init_state_grow
from jax_morph.simulation import simulation

# state update functions
from jax_morph.division_and_growth.cell_division import S_cell_division
from jax_morph.division_and_growth.cell_growth import S_grow_cells

from jax_morph.mechanics import morse
from jax_morph.chemicals.secdiff import S_ss_chemfield

from jax_morph.cell_internals.divrates import S_set_divrate, div_nn
from jax_morph.cell_internals.secretion import sec_nn
from jax_morph.cell_internals.grad_estimate import S_chemical_gradients
from jax_morph.cell_internals.hidden_state import hidden_state_nn, S_hidden_state




### CREATE DEFAULT PARAMS ###
def default_params(key):


    # Initialization and number of added cells. 
    ncells_init = 1 
    ncells_add = 149


    n_chem = 2
    hidden_state_size = 32
    hidden_state_decay = .8


    HID_HIDDEN = [128]
    DIV_HIDDEN = []
    SEC_HIDDEN = []


    #--------------------------#
    #   MORE STABLE PARAMETERS #
    #--------------------------#

    n_dim = 2
    n_celltype = 1


    ### CELL DIMENSION

    cellRad = .5
    cellRadBirth = float(cellRad / np.sqrt(2))


    ### DIFFUSION

    key, subkey = split(key)

    diffCoeff = 2*jax.random.uniform(subkey, (n_chem,))#np.ones(n_chem)
    degRate = np.ones(n_chem)

    r_cutoffDiff = float(np.log(10)/diffCoeff.max())
    r_onsetDiff = max(r_cutoffDiff - .3, 0.)


    ### SECRETION

    key, subkey = split(key)
    sec_max = 5*jax.random.uniform(subkey, (n_chem,)) 

    #rows are ctypes, cols are chemicals
    ctype_sec_chem = np.ones((n_celltype,n_chem), dtype=np.int16)


    # MORSE POTENTIAL

    alpha = 2.7
    epsilon = 3.

    r_cutoff = 2.2*cellRad
    r_onset = r_cutoff - .2

    mech_relaxation_steps = 15



    #--------------------------#
    #   INIT DICTIONARIES      #
    #--------------------------#

    train_params = {
        'n_dim':                    False,
        'n_chem':                   False,
        'ctype_sec_chem':           False,
        'hidden_state_size':        False,
        'hidden_state_decay':       False,
        
        'sec_max':                      True,
        
        'cellRad' :                 False,
        'cellRadBirth' :            False,
        
        'diffCoeff' :                   True,
        'degRate' :                 False,
        'r_onsetDiff' :             False,
        'r_cutoffDiff' :            False,
        
        'alpha':                    False,
        'epsilon':                  False,
        'r_onset' :                 False,
        'r_cutoff' :                False,
        'mech_relaxation_steps' :   False,
        
        'ncells_init' :             False,
        'ncells_add':               False,

        'hid_hidden':               False,
        'div_hidden':               False,
        'sec_hidden':               False,
    }


    params = {
        'n_dim':                    n_dim,
        'n_chem':                   n_chem,
        'ctype_sec_chem':           ctype_sec_chem,
        'hidden_state_size':        hidden_state_size,
        'hidden_state_decay':       _maybe_array('hidden_state_decay', hidden_state_decay, train_params),
        
        'sec_max':                  sec_max,
        
        'cellRad' :                 cellRad,
        'cellRadBirth' :            cellRadBirth,
        
        'diffCoeff' :               diffCoeff,
        'degRate' :                 degRate,
        'r_onsetDiff' :             r_onsetDiff,
        'r_cutoffDiff' :            r_cutoffDiff,
        
        'alpha':                    _maybe_array('alpha', alpha, train_params),
        'epsilon':                  _maybe_array('epsilon', epsilon, train_params),
        'r_onset' :                 r_onset,
        'r_cutoff' :                r_cutoff,
        'mech_relaxation_steps' :   mech_relaxation_steps,
        
        'ncells_init' :             ncells_init,
        'ncells_add':               ncells_add,

        'hid_hidden':               HID_HIDDEN,
        'div_hidden':               DIV_HIDDEN,
        'sec_hidden':               SEC_HIDDEN,
    }

    return params, train_params



### CREATE CELLSTATE DATASTRUCTURE ###
@jdc.dataclass
class CellState:
    '''
    Dataclass containing the system state.

    STATE
    -----

    '''

    # STATE
    position:   jax_md.util.Array
    celltype:   jax_md.util.Array
    radius:     jax_md.util.Array
    chemical:   jax_md.util.Array
    chemgrad:   jax_md.util.Array
    hidden_state: jax_md.util.Array
    divrate:    jax_md.util.Array
    key:        jax_md.util.Array


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
            'hidden_state' : np.empty(shape=(0, hidden_size),       dtype=np.float32),
            'divrate'   :   np.empty(shape=(0,),                    dtype=np.float32),
            'key'       :   None,
        }


        return cls(**defaultstate)
    



def build_sim_from_params(params, train_params, key):

    N_CELLS_INIT = params['ncells_init']

    HID_HIDDEN = params['hid_hidden']
    DIV_HIDDEN = params['div_hidden']
    SEC_HIDDEN = params['sec_hidden']

    HID_STATE_DECAY = params['hidden_state_decay']


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
                                radius=            True, 
                                chemical=          True,
                                chemgrad=          True,
                                divrate=           True,
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
                                divrate=       False,
                                hidden_state=      True,
                                key=           False
                                )

    div_init, div_nn_apply = div_nn(params,
                                    train_params,
                                    DIV_HIDDEN,
                                    use_state_fields,
                                    train=True,
                                    w_init=hk.initializers.Constant(0.),
                                    transform_mlp_out=jax.nn.softplus
                                    )

    key, init_key = split(key)
    params, train_params = div_init(istate, init_key)


    ### ANN: Secretion
    use_state_fields = CellState(position=      False, 
                                celltype=      False, 
                                radius=        False, 
                                chemical=      False,
                                chemgrad=      False,
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
                                    )

    key, init_key = split(key)
    params, train_params = sec_init(istate, init_key)



    ### Simulation loop
    fstep = [
        # ENV CHANGES
        S_cell_division,
        S_grow_cells,
        morse.S_mech_morse_relax,
        partial(S_ss_chemfield, sec_fn=sec_nn_apply, n_iter=3),

        # SENSING
        S_chemical_gradients,

        # INTERNAL (HIDDEN) STATE
        partial(S_hidden_state, dhidden_fn=eqx.filter_jit(hid_nn_apply), state_decay=HID_STATE_DECAY),

        # POLICIES
        partial(S_set_divrate, divrate_fn=eqx.filter_jit(div_nn_apply))
    ]


    #sim_init, sim_step = simulation(fstep, params, fspace)

    SimulBlocks = namedtuple('SimulBlocks', ['fstep', 'fspace', 'istate', 'params', 'train_params'])

    return SimulBlocks(fstep, fspace, istate, params, train_params) #, sim_init, sim_step







### TRAIN FUNCTION ###
def train(key, 
          EPOCHS, 
          EPISODES_PER_UPDATE, 
          EPISODES_PER_EVAL, 
          LEARNING_RATE, 
          train_loss, 
          eval_loss,
          fstep, 
          fspace, 
          istate,
          params,
          train_params,
          normalize_grads=False,
          optimizer=optax.adam,
          save_every=10,
          save_grads=False,
          ):

    p, hp = eqx.partition(params, train_params)

    # init optimizer
    optimizer = optimizer(LEARNING_RATE)
    opt_state = optimizer.init(p)


    #--------------------------------------------
    #store loss at initial params and calc grad 

    key, *batch_subkeys = split(key, EPISODES_PER_UPDATE+1)
    batch_subkeys = np.array(batch_subkeys)

    ll, grads = value_and_grad(avg_loss)(p, hp, train_loss, batch_subkeys, fstep=fstep, fspace=fspace, istate=istate)


    key, *eval_subkeys = split(key, EPISODES_PER_EVAL+1)
    eval_subkeys = np.array(eval_subkeys)

    l = avg_loss(p, hp, eval_loss, eval_subkeys, fstep=fstep, fspace=fspace, istate=istate)

    #store initial params and loss
    loss_t = [float(l)]
    params_t = [p]
    grads_t = [grads] if save_grads else None

    #--------------------------------------------

    pbar = trange(EPOCHS, desc='Loss: {:.4f}'.format(l))
    for t in pbar:
        
        try:
            #generate batch of random keys
            key, *batch_subkeys = split(key, EPISODES_PER_UPDATE+1)
            batch_subkeys = np.array(batch_subkeys)
        
            #normalize grads
            if normalize_grads:
                grads = jax.tree_map(lambda x: x/(np.linalg.norm(x)+1e-10), grads)


            # sgd step
            updates, opt_state = optimizer.update(grads, opt_state, p)

            p = eqx.apply_updates(p, updates)
        
            #clip diffCoeff if trained
            if None != p['diffCoeff']:
                p['diffCoeff'] = np.clip(p['diffCoeff'],.2)
        
            #estimate actual avg loss
            key, *eval_subkeys = split(key, EPISODES_PER_EVAL+1)
            eval_subkeys = np.array(eval_subkeys)

            l = avg_loss(p, hp, eval_loss, eval_subkeys, fstep=fstep, fspace=fspace, istate=istate)
        
            # surrogate loss and grads
            ll, grads = value_and_grad(avg_loss)(p, hp, train_loss, batch_subkeys, fstep=fstep, fspace=fspace, istate=istate)

            loss_t += [float(l)]


            #store
            if t == EPOCHS-1:
                params_t += [p]
                if save_grads:
                    grads_t += [grads]
                    
            elif t%save_every==0:
                params_t += [p]
                if save_grads:
                    grads_t += [grads]


            pbar.set_description('Loss: {:.4f}'.format(l))

        except FloatingPointError:
            print('NaN or Overflow')
            break

        except KeyboardInterrupt:
            print('Interrupted')
            break


    return loss_t, params_t, grads_t




### OPTIMIZATION FUNCTION ###
def run_optimization(train_key, 
                     sim,
                     metric_fn,
                     metric_type,
                     epochs=10, 
                     episodes_per_update=4, 
                     episodes_per_eval=64, 
                     learning_rate=1e-3, 
                     optimizer=optax.adam,
                     reinforce_gamma=.95,
                     save_every=10,
                     save_grads=False,
                     normalize_grads=True,
                     ):



    train_loss = eqx.filter_jit(partial(loss, metric_fn=metric_fn, metric_type=metric_type, REINFORCE=True, GAMMA=reinforce_gamma))
    eval_loss = eqx.filter_jit(partial(loss, metric_fn=metric_fn, REINFORCE=False))


    loss_t, params_t, grads_t = train(train_key, 
                                    epochs, 
                                    episodes_per_update, 
                                    episodes_per_eval, 
                                    learning_rate, 
                                    train_loss, 
                                    eval_loss,
                                    sim.fstep,
                                    sim.fspace,
                                    sim.istate,
                                    sim.params,
                                    sim.train_params,
                                    normalize_grads=normalize_grads,
                                    optimizer=optimizer,
                                    save_every=save_every,
                                    save_grads=save_grads,
                                    )
    
    return loss_t, params_t, grads_t