import jax
import jax.numpy as np
from jax import value_and_grad, vmap, random
from jax.random import split

import optax
import equinox as eqx
import haiku as hk

import jax_md
import jax_md.dataclasses as jdc
from jax_md import space

from tqdm import trange, tqdm
from functools import partial
from collections import namedtuple

import imageio, os, shutil

import matplotlib.pyplot as plt
from Ramya.mech_homogeneous_growth.chemical import S_fixed_chemfield

### JAX-MORPH IMPORTS ###

from jax_morph.optimization.losses import avg_loss, loss
from jax_morph.utils import logistic
from jax_morph.datastructures import SpaceFunc
from jax_morph.utils import _maybe_array

#from jax_morph.initial_states import init_state_grow
from jax_morph.simulation import simulation, sim_trajectory

# state update functions
from jax_morph.division_and_growth.cell_division import S_cell_division
from jax_morph.division_and_growth.cell_growth import S_grow_cells

from jax_morph.mechanics import morse
from jax_morph.mechanics.morse import S_mech_morse_relax

from jax_morph.chemicals.secdiff import S_ss_chemfield

from jax_morph.cell_internals.divrates import S_set_divrate, div_nn
from jax_morph.cell_internals.stress import S_set_stress
from jax_morph.cell_internals.secretion import sec_nn
from jax_morph.cell_internals.grad_estimate import S_chemical_gradients
from jax_morph.cell_internals.hidden_state import hidden_state_nn, S_hidden_state
from jax_morph.visualization import draw_circles_ctype
from jax_morph.chemicals.diffusion import diffuse_allchem_ss_exp, diffuse_allchem_ss


import jax
from jax import lax


# Modify to have random amounts in gene_vec
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
    gene_vec = state.gene_vec.at[0].set(.1 + state.gene_vec[0])
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

def synnet(params, train_params, use_state_fields=None, train=True):

    if use_state_fields is None:
        raise ValueError('Input fields flags must be passed explicitly as a CellState dataclass.')
    def _nn(x):
        mlp = hk.nets.MLP([x.shape[1]],
                          activation=jax.nn.leaky_relu,)
        
        return mlp(x)
    _nn = hk.without_apply_rng(hk.transform(_nn))

    def _circuit_solve(x0, n_steps, W, k, I, dt=.1):
        def _step(xt,t):
            x_dot = jax.nn.sigmoid(xt.dot(W))-k*xt + I
            xtt = xt + x_dot*dt
            return xtt, xtt
        x, _ = lax.scan(_step, x0, np.arange(n_steps))
        return x


    # Initialize gene interactions
    def init(state, key):     
        in_fields = np.hstack([f if len(f.shape)>1 else f[:,np.newaxis] for f in jax.tree_leaves(eqx.filter(state, use_state_fields))])
        n_genes = in_fields.shape[1] + params["hidden_genes"] + params["n_chem"] + 1
        # Initialize weights matrix for gene interactions of n_genes x n_genes
        p = np.zeros((n_genes, n_genes))
        #p = random.normal(key, shape=(n_genes, n_genes), dtype=np.float32)*0.1
        # Add to param dict
        params['gene_fn'] = p
        params['k'] = np.ones((n_genes))

        # no need to update train_params when generating initial state
        if type(train_params) is dict:
            #set trainability flag
            train_p = jax.tree_map(lambda x: train, p)
            train_params['gene_fn'] = train_p
            train_params['k'] = False
            return params, train_params
        
        else:
            return params
            
        
    # Use gene interactions to update gene vector    
    def fwd(state, params):
        in_fields = np.hstack([f if len(f.shape)>1 else f[:,np.newaxis] for f in jax.tree_leaves(eqx.filter(state, use_state_fields))])
        input_vec = np.hstack((in_fields, np.zeros((in_fields.shape[0], state.gene_vec.shape[1] - in_fields.shape[1]))))
        gene_vec = _circuit_solve(x0=state.gene_vec, n_steps=100, W=params['gene_fn'], k=params['k'], I=input_vec)
        return gene_vec
        
    return init, fwd

### CREATE DEFAULT PARAMS ###
def default_params(key, n_chem=2):
    '''
    To change n_chem you need to rebuild all the params.
    '''

    

    # Initialization and number of added cells. 
    ncells_init = 1 
    ncells_add = 149


    hidden_genes = 150



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

    diffCoeff = 2.*jax.random.uniform(subkey, (n_chem,))#np.ones(n_chem)
    degRate = np.ones(n_chem)

    r_cutoffDiff = float(np.log(10)/diffCoeff.max())
    r_onsetDiff = max(r_cutoffDiff - .3, 0.)


    ### SECRETION

    key, subkey = split(key)
    #sec_max = 5*jax.random.uniform(subkey, (n_chem,))
    sec_max = 2.*np.ones((n_chem,)) 


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
        'hidden_genes':        False,
        
        'sec_max':                      False,
        
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
    }


    params = {
        'n_dim':                    n_dim,
        'n_chem':                   n_chem,
        'ctype_sec_chem':           ctype_sec_chem,
        'hidden_genes':             hidden_genes,        
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
        'ncells_add':               ncells_add
    }

    return params, train_params


### CREATE CELLSTATE DATASTRUCTURE ###
import jax_md
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
    field:      jax_md.util.Array
    gene_vec:   jax_md.util.Array
    stress:     jax_md.util.Array
    divrate:    jax_md.util.Array
    key:        jax_md.util.Array


    @classmethod
    def default_init(cls, n_dim=2, n_chem=1, n_inputs=9, hidden_genes=10):
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
            'field'     :   np.empty(shape=(0,),                    dtype=np.float32),
            # Size of gene vec: # of inputs, # of hidden genes, # of outputs (divisions, secretion)
            'gene_vec'  :   np.empty(shape=(0, hidden_genes + n_inputs + n_chem + 1), dtype=np.float32),
            'stress'    :   np.empty(shape=(0,),       dtype=np.float32),
            'divrate'   :   np.empty(shape=(0,),                    dtype=np.float32),
            'key'       :   None,
        }


        return cls(**defaultstate)

def sec_fn(state, params):
    sec = state.gene_vec[:, -params["n_chem"]:]*params['sec_max']
    ctype_sec_chem = np.vstack((np.zeros(params['n_chem']), params['ctype_sec_chem']))
    @vmap
    def sec_mask(ct):
        return ctype_sec_chem[np.int16(ct)] #change if we switch to dead cells = -1
        
    mask = sec_mask(state.celltype)
    return sec*mask
def div_fn(state, params, transform_fwd=None):
    if transform_fwd is None:
        transform_fwd = lambda x, y: y
    div = state.gene_vec[:, -params["n_chem"]-1]
    div = div*logistic(state.radius+.06, 50, params['cellRad'])
    div = np.where(state.celltype<1.,0,div)
    div = transform_fwd(state, div)
    return div
def S_ss_gene_vec(state, params, fspace, syn_fun=None):
    gene_vec = syn_fun(state, params)
    state = jdc.replace(state, gene_vec=gene_vec)
    return state


def build_sim_from_params(params, train_params, key, n_inputs=9, div_fwd=None):

    N_CELLS_INIT = params['ncells_init']

    ## Initialize CellState
    fspace = SpaceFunc(*space.free())
    istate = CellState.default_init(n_dim=params['n_dim'], 
                                    n_chem=params['n_chem'],
                                    hidden_genes=params['hidden_genes'],
                                    n_inputs=n_inputs,
                                    )
    key, init_key = split(key)
    istate = init_state_grow(init_key, istate, params, fspace, N_CELLS_INIT)
    use_state_fields = CellState(position=      False, 
                                celltype=      False, 
                                radius=            True, 
                                chemical=          True,
                                chemgrad=          True,
                                field=             False,
                                divrate=           True,
                                gene_vec=  False,
                                stress=    True,
                                key=           False
                                )
    synnet_init, synnet_apply = synnet(params, train_params, use_state_fields)
    params, train_params = synnet_init(istate, key)


    ### Simulation loop
    fstep = [
            # ENV CHANGES
            S_cell_division,
            S_grow_cells,
            morse.S_mech_morse_relax,
            partial(S_ss_gene_vec, syn_fun=synnet_apply),
            partial(S_ss_chemfield, sec_fn=sec_fn, diffusion_fn=diffuse_allchem_ss_exp, n_iter=1),
            # SENSING
            S_fixed_chemfield,
            S_chemical_gradients,
            S_set_stress,
            # POLICIES
            partial(S_set_divrate, divrate_fn=eqx.filter_jit(partial(div_fn, transform_fwd=div_fwd)))
        ]

    SimulBlocks = namedtuple('SimulBlocks', ['fstep', 'fspace', 'istate', 'params', 'train_params'])
    return SimulBlocks(fstep, fspace, istate, params, train_params) #, sim_init, sim_step


# Regularized loss function
@eqx.filter_jit
@eqx.filter_vmap(default=None, kwargs=dict(sim_key=0))
def loss(params, 
         hyper_params,
         fstep,
         fspace,
         istate,
         sim_key=None,
         metric_fn=None,
         metric_type='reward',
         REINFORCE=True,
         GAMMA=.99,
         ncells_add=None,
         LAMBDA=0.0
         ):
    '''
    Reinforce loss on trajectory (with discounting). Rewards are differences in successive state metrics.

    If REINFORCE=False, then the loss is just the state measure on the final state.

    GAMMA is the discount factor for the calculation of the returns.

    If metric_type='reward', it is maximized, if metric_type='cost', it is minimized.

    '''

    #simulation length
    ncells_add = hyper_params['ncells_add'] if ncells_add is None else ncells_add
    
    def _sim_trajectory(istate, sim_init, sim_step, ncells_add, key=None):

        state = sim_init(istate, ncells_add, key)

        def scan_fn(state, i):
            state, logp = sim_step(state)
            measure = metric_fn(state)
            return state, (logp, measure)


        iterations = np.arange(ncells_add)
        fstate, aux = jax.lax.scan(scan_fn, state, iterations)

        return fstate, aux

    # merge params dicts
    all_params = eqx.combine(params, hyper_params)

    #forward pass - simulation
    sim_init, sim_step = simulation(fstep, all_params, fspace)
    _, (logp, measures) = _sim_trajectory(istate, sim_init, sim_step, ncells_add, sim_key)

    
    if REINFORCE:
        
        def _returns_rec(rewards):
            Gs=[]
            G=0
            for r in rewards[::-1]:
                G = r+G*GAMMA
                Gs.append(G)

            return np.array(Gs)[::-1]
        
        
        measures = np.append(np.array([metric_fn(istate)]),measures)
        
        if metric_type=='reward':
            rewards = np.diff(measures)
        elif metric_type=='cost':
            rewards = -np.diff(measures)


        returns = _returns_rec(rewards)

        # standardizing returns helps with convergence
        returns = (returns-returns.mean())/(returns.std()+1e-8)

        loss = -np.sum(logp*jax.lax.stop_gradient(returns))
        if LAMBDA > 0.:
            if "gene_fn" in params:
                loss += LAMBDA*np.abs(params["gene_fn"]).sum()
        return loss

    else:
        return measures[-1]

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
                    
            elif t%save_every==0 and t>0:
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
                     LAMBDA=0.0,
                     reinforce=True,
                     ):



    train_loss = eqx.filter_jit(partial(loss, metric_fn=metric_fn, metric_type=metric_type, REINFORCE=reinforce, GAMMA=reinforce_gamma, LAMBDA=LAMBDA))
    eval_loss = eqx.filter_jit(partial(loss, metric_fn=metric_fn, REINFORCE=False, LAMBDA=LAMBDA))


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



def mask_metric(mask_fn=None, reward=3., penalty=-1., xasym_penalty=.5):
        
    def metric(state):
        
        alive = state.celltype > 0
        n = np.sum(alive)
        
        mask = mask_fn(state.position)

        m = np.sum(np.where(mask, reward, penalty)*alive)

        # penalize asymmetric growth
        m -= xasym_penalty*np.abs(np.sum(state.position[:, 0] * alive))
            
        return m
    
    return metric

def v_mask(pos):
    '''
    Constrain cell growth in a V shape.
    '''
    return (pos[:,1]+1.5 > .5*np.abs(pos[:,0])) * (pos[:,1]+1.5 < 3.5+.5*np.abs(pos[:,0])) * (pos[:,1]>0.)


v_metric = mask_metric(v_mask)

def position_sum_of_squares(state, coordinate=1):

    alive = state.celltype > 0

    m = np.sum((state.position[:, coordinate] * alive)**2)

    m = m / np.sum(alive)

    return m
    
def cv_divrates(state):

    return np.std(state.divrate)/(np.mean(state.divrate) + 1e-6)