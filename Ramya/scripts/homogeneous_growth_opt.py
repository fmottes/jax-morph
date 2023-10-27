import sys
sys.path.append('../')
sys.path.append('/n/home10/rdeshpande/morphogenesis/jax-morph')
ROOT_DIR = 'n/home10/rdeshpande/morphogenesis/data/paper/'


import jax.numpy as np
from jax import random, tree_map, value_and_grad
from jax.nn import tanh, softplus, sigmoid
from jax_md import space, util
import jax_md.dataclasses as jdc
from jax_morph.optimization.losses import loss, avg_loss

########## IMPORT JAX-MORPH FUNCTIONS ##########
################################################

from jax_morph.datastructures import SpaceFunc
from jax_morph.utils import _maybe_array, logistic
from jax_morph.simulation import simulation

# IMPORT STATE-CHANGING FUNCTIONS
from jax_morph.division_and_growth.cell_division import S_cell_division
from jax_morph.division_and_growth.cell_growth import S_grow_cells

from jax_morph.mechanics.morse import S_mech_morse_relax
from jax_morph.cell_internals.stress import S_set_stress
from jax_morph.chemicals.secdiff import S_ss_chemfield

from jax_morph.cell_internals.divrates import S_set_divrate, div_nn
from jax_morph.cell_internals.secretion import sec_nn
from jax_morph.cell_internals.grad_estimate import S_chemical_gradients
from jax_morph.cell_internals.hidden_state import hidden_state_nn, S_hidden_state

from jax_morph.initial_states import init_state_grow
from Ramya.mech_homogeneous_growth.chemical import S_fixed_chemfield


from functools import partial
import equinox as eqx
import haiku as hk
import optax

# For saving data
import pickle
from pathlib import Path



########## DEFINE PARAMETERS ##########
################################################
# Define parameters --blue particles are type 1, orange are type 2
# keep type casting to place vars in gpu memory

# Number of chemical signals
n_chem = 2


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

sec_max = np.ones((n_chem,), dtype=np.float32)
ctype_sec_chem = np.ones((1, 2))

# MORSE POTENTIAL
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
    'hidden_state_size': False
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
    
    # build space handling function and initial state
key = random.PRNGKey(0)
fspace = SpaceFunc(*space.free())

N_CELLS_INIT = params['ncells_init']



#generate empty data structure with correct shapes
istate = CellState.default_init(n_dim=params['n_dim'], 
                                n_chem=params['n_chem'],
                                hidden_size=params['hidden_state_size']
                                )

# populate initial state by growing from single cell
key, init_key = random.split(key)
istate = init_state_grow(init_key, istate, params, fspace, N_CELLS_INIT)

#randomly initialize hidden states
key, init_key = random.split(key)
# hidden_regulation_init = 5*jax.random.normal(init_key, shape=istate.hidden_state.shape)
hidden_state_init = softplus(10*(random.uniform(init_key, shape=istate.hidden_state.shape)*2 - 1))
istate = jdc.replace(istate, hidden_state=hidden_state_init)

# randomly initialize chemical species
key, init_key = random.split(key)
ichem = random.uniform(init_key, istate.chemical.shape)*params['sec_max']
istate = jdc.replace(istate, chemical=ichem)

#hidden neurons per layer
HID_HIDDEN = 8

#input fields to the network
use_state_fields = CellState(position=      False, 
                             celltype=      False, 
                             radius=            False, 
                             chemical=          True,
                             chemgrad=          True,
                             field=             False,
                             stress=            True,
                             divrate=           False,
                             hidden_state=      False,
                             key=           False
                            )


# init nn functions
hid_init, hid_nn_apply = hidden_state_nn(params,
                                         train_params,
                                         HID_HIDDEN,
                                         use_state_fields,
                                         train=True,
                                         transform_mlp_out=tanh,
                                         )


key, init_key = random.split(key)
params, train_params = hid_init(istate, init_key)

#hidden neurons per layer

DIV_HIDDEN = []
transform_mlp_out=sigmoid

#input fields to the network
use_state_fields_div = CellState(position=   False, 
                             celltype=   False, 
                             radius=     False, 
                             chemical=     False,
                             field=      False,
                             stress=    False,
                             chemgrad=   False,
                             hidden_state= True,
                             divrate=    False, 
                             key=        False
                            )
transform_fwd = lambda state, divrates: divrates*state.field
#transform_fwd=None
# init nn functions
div_init, div_nn_apply = div_nn(params,
                                train_params,
                                DIV_HIDDEN,
                                use_state_fields_div,
                                train=True,
                                w_init=hk.initializers.Constant(0.0),
                                transform_mlp_out=sigmoid,
                                transform_fwd=transform_fwd,)

#initialize network parameters
key, init_key = random.split(key)
params, train_params = div_init(istate, init_key)

#hidden neurons per layer
SEC_HIDDEN = []


#input fields to the network
use_state_fields_sec = CellState(position=   False, 
                             celltype=   False, 
                             radius=     False, 
                             chemical=      False,
                             chemgrad=   False,
                             field=      False,
                             stress=   False,
                             divrate=    False,
                             hidden_state= True, 
                             key=        False
                            )


# init nn functions
sec_init, sec_nn_apply = sec_nn(params,
                                train_params,
                                SEC_HIDDEN,
                                use_state_fields_sec,
                                w_init=hk.initializers.Constant(0.0),
                                train=True)


#initialize network parameters
key, init_key = random.split(key)
params, train_params = sec_init(istate, init_key)


# functions in this list will be executed in the given order
# at each simulation step

fstep = [
    # ENV CHANGES
    S_cell_division,
    #S_cell_div_indep_MC,
    S_grow_cells,
    partial(S_mech_morse_relax, dt=.0001),
    partial(S_ss_chemfield, sec_fn=sec_nn_apply, n_iter=3),

    # SENSING
    #chemicals sensed directly
    S_chemical_gradients,
    S_fixed_chemfield,
    S_set_stress,
    # INTERNAL (HIDDEN) STATE
    #no hidden state in this case
    # INTERNAL (HIDDEN) STATE
    partial(S_hidden_state, dhidden_fn=eqx.filter_jit(hid_nn_apply), state_decay=.0),
    # POLICIES
    partial(S_set_divrate, divrate_fn=eqx.filter_jit(div_nn_apply))
]


sim_init, sim_step = simulation(fstep, params, fspace)

def cv_divrates(state):
    ''' 
    Coefficient of variation of division rates loss.
    '''
    return np.power(np.std(state.divrate)/np.mean(state.divrate), 2)


def train(key,
          params, train_params, 
          EPOCHS, 
          EPISODES_PER_UPDATE, 
          EPISODES_PER_EVAL, 
          LEARNING_RATE, 
          rloss, 
          sloss, 
          fstep, 
          fspace, 
          istate,
          normalize_grads=True,
          ):

    p, hp = eqx.partition(params, train_params)

    # init optimizer
    optimizer = optax.adam(LEARNING_RATE)
    opt_state = optimizer.init(p)


    #--------------------------------------------
    #store loss at initial params and calc grad 

    key, *batch_subkeys = random.split(key, EPISODES_PER_UPDATE+1)
    batch_subkeys = np.array(batch_subkeys)

    ll, grads = value_and_grad(avg_loss)(p, hp, rloss, batch_subkeys, fspace=fspace, fstep=fstep, istate=istate)


    key, *eval_subkeys = random.split(key, EPISODES_PER_EVAL+1)
    eval_subkeys = np.array(eval_subkeys)

    l = avg_loss(p, hp, sloss, eval_subkeys, fstep=fstep, fspace=fspace, istate=istate)
    print(float(l))
    #store initial params and loss
    loss_t = [float(l)]
    params_t = [p]
    grads_t = [grads]

    #--------------------------------------------


    for t in range(EPOCHS):
        #generate batch of random keys
        key, *batch_subkeys = random.split(key, EPISODES_PER_UPDATE+1)
        batch_subkeys = np.array(batch_subkeys)
        #normalize grads
        if normalize_grads:
            grads = tree_map(lambda x: x/(np.linalg.norm(x)+1e-10), grads)
        # sgd step
        updates, opt_state = optimizer.update(grads, opt_state, p)

        p = eqx.apply_updates(p, updates)
    
        #clip diffCoeff if trained
        if None != p['diffCoeff']:
            p['diffCoeff'] = np.clip(p['diffCoeff'],.2)
    
        #estimate actual avg loss
        key, *eval_subkeys = random.split(key, EPISODES_PER_EVAL+1)
        eval_subkeys = np.array(eval_subkeys)

        l = avg_loss(p, hp, sloss, eval_subkeys, fstep=fstep, fspace=fspace, istate=istate)
    
        # surrogate loss and grads
        ll, grads = value_and_grad(avg_loss)(p, hp, rloss, batch_subkeys, fstep=fstep, fspace=fspace, istate=istate)


        #store
        loss_t += [float(l)]
        params_t += [p]
        grads_t += [grads]   
        print(float(l))
    return loss_t, params_t, grads_t



# First learn homogeneous growth parameters.
def optimize(key, params, train_params, istate, EPOCHS, EPISODES_PER_UPDATE, EPISODES_PER_EVAL, LEARNING_RATE):
    rloss = eqx.filter_jit(partial(loss, metric_fn=cv_divrates, REINFORCE=False, metric_type='cost', GAMMA=0.95))
    sloss = eqx.filter_jit(partial(loss, metric_fn=cv_divrates, metric_type='cost', REINFORCE=False))    
    loss_t, params_t, grads_t = train(key, params, train_params, EPOCHS, EPISODES_PER_UPDATE, EPISODES_PER_EVAL, LEARNING_RATE, rloss, sloss, fstep, fspace, istate)
    return loss_t, params_t, grads_t

def main():
    print("in main loop")
    NUM_OPT = int(sys.argv[1])
    EPOCHS = int(sys.argv[2])
    EPISODES_PER_UPDATE = int(sys.argv[3])
    EPISODES_PER_EVAL = int(sys.argv[4])
    LEARNING_RATE = float(sys.argv[5])
    key = random.PRNGKey(0)
    params_tt, loss_tt, grads_tt = [], [], []
    for i in range(NUM_OPT):
        print("loop %s" % i)
        _, key = random.split(key)
        loss_i, params_i, grads_i = optimize(key, params, train_params, istate, EPOCHS, EPISODES_PER_UPDATE, EPISODES_PER_EVAL, LEARNING_RATE)
        params_tt.append(params_i)
        loss_tt.append(loss_i)
        grads_tt.append(grads_i)

    pickle.dump(params_tt, open(ROOT_DIR + "homogeneous_growth_" + str(NUM_OPT) + "_hidden_" + str(HID_HIDDEN) + "_params_power_law", 'wb'))
    pickle.dump(loss_tt, open(ROOT_DIR + "homogeneous_growth_" + str(NUM_OPT) + "_hidden_" + str(HID_HIDDEN) + "_loss_power_law", 'wb'))
    pickle.dump(grads_tt, open(ROOT_DIR + "homogeneous_growth_" + str(NUM_OPT) + "_hidden_" + str(HID_HIDDEN) + "_grads_power_law", 'wb'))

if __name__ == "__main__":
    main()
