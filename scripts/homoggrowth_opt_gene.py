import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
from functools import partial
import pickle
import sys
sys.path.append("../")
sys.path.append('/n/home10/rdeshpande/morphogenesis/jax-morph')
ROOT_DIR = '/n/holylabs/LABS/brenner_lab/Lab/jxm_data/'

import jax
import jax.numpy as np
import jax.tree_util as jtu

key = jax.random.PRNGKey(1)

jax.config.update("jax_debug_nans", True)
jax.config.update("jax_enable_x64", True)

import jax_md
import equinox as eqx

import jax_morph as jxm
import optax

class CellState(jxm.BaseCellState):
    chemical:           jax.Array
    secretion_rate:     jax.Array
    chemical_grad:      jax.Array
    hidden_state:       jax.Array
    mechanical_stress:  jax.Array
    field:              jax.Array


class ChemicalField(jxm.SimulationStep):
    def return_logprob(self) -> bool:
        return False
    @jax.named_scope("jax_morph.ChemicalField")
    def __call__(self, state, *, key=None, **kwargs):
        center = np.average(state.position, axis=0)
        chemfield_disp = jax.vmap(state.displacement, (0, None))(state.position, center)
        chemfield_disp = np.linalg.norm(chemfield_disp, axis=1)
        chemfield = 1.0/(2 + .4*np.power(chemfield_disp, 2))
        chemfield = np.where(state.celltype.sum(1)[:,None] > 0, chemfield[:, np.newaxis], 0.) 
        state = eqx.tree_at(lambda s: s.field, state, chemfield)
        return state

@eqx.filter_jit
def reinforce_loss(model, istate, *, cost_fn, key, n_steps=1, BATCH_SIZE=1, GAMMA=.9, LAMBDA=.001):

    subkeys = jax.random.split(key, BATCH_SIZE)
    subkeys = np.asarray(subkeys)

    vsim = jax.vmap(partial(jxm.simulate, history=True), (None, None, 0, None))
    trajectory, logp = vsim(model, istate, subkeys, n_steps)


    istate = jtu.tree_map(lambda x: np.repeat(x[None,None,:,:],BATCH_SIZE,0), istate)

    trajectory = jtu.tree_map(lambda *v: np.concatenate(v,1), *[istate, trajectory])


    cost = jax.vmap(cost_fn)(trajectory)


    #discounted costs
    def _returns_rec(rewards):
        Gs=[]
        G=0
        for r in rewards[::-1]:
            G = r+G*GAMMA
            Gs.append(G)

        return np.array(Gs)[::-1]
    
    
    #cost = jax.vmap(_returns_rec)(cost)

    #cost = (cost-cost.mean(-1, keepdims=True))/(cost.std(-1, keepdims=True)+1e-8)


    #no - sign because we assume a cost instead of a reward
    #loss = np.sum(jax.lax.stop_gradient(cost)*logp)
    loss = np.sum(cost)

    #L1 penalty on weights
    reg = jax.tree_map(lambda x: np.abs(x).sum(), eqx.filter(model, eqx.is_array))
    reg = jax.tree_util.tree_reduce(lambda x,y: x+y, reg)

    return loss + LAMBDA*reg

def division_cv(trajectory):
    cost = np.std(trajectory.division[-1, :])/np.mean(trajectory.division[-1, :])
    return cost

def run_experiment():
    key = jax.random.PRNGKey(1)
    N_OPT = int(sys.argv[1])
    EPOCHS = int(sys.argv[2])
    BATCH_SIZE = int(sys.argv[3])
    N_HIDDEN = int(sys.argv[4]) #32
    N_CHEM = int(sys.argv[5]) #10
    LEARNING_RATE = 1e-2
    COST_FN = division_cv
    LAMBDA=0.

    N_DIM = 2
    N_CTYPES = 1
    
    N = 100
    N_ADD = int(N-1)

    disp, shift = jax_md.space.free()
    istate = CellState(
        displacement=   disp,
        shift=          shift,
        position=       np.zeros(shape=(N,N_DIM)),
        celltype=       np.zeros(shape=(N,N_CTYPES)).at[0].set(1.),
        radius=         np.zeros(shape=(N,1)).at[0].set(.5),
        division=       np.zeros(shape=(N,1)).at[0].set(1.),
        chemical=       np.zeros(shape=(N,N_CHEM)),
        chemical_grad=  np.zeros(shape=(N,int(N_DIM*N_CHEM))),
        secretion_rate= np.zeros(shape=(N,N_CHEM)).at[0].set(1.),
        field=          np.zeros(shape=(N,1)),
        mechanical_stress=np.zeros(shape=(N,1)),
        hidden_state=   np.zeros(shape=(N,N_HIDDEN))
    )
    
    key, init_key = jax.random.split(key)
    mech_potential = jxm.env.mechanics.MorsePotential(epsilon=3., alpha=2.8)

    model = jxm.Sequential(
    substeps=[
        jxm.env.CellDivision(),
        ChemicalField(),
        jxm.env.CellGrowth(growth_rate=.03, max_radius=.5, growth_type='linear'),
        jxm.env.mechanics.SGDMechanicalRelaxation(mech_potential),
        jxm.cell.sensing.LocalMechanicalStress(mech_potential),
        jxm.env.SteadyStateDiffusion(degradation_rate=.1, diffusion_coeff=.2),
        jxm.cell.sensing.LocalChemicalGradients(),
        jxm.cell.GeneNetwork(istate,
                            input_fields=['chemical', 'chemical_grad','mechanical_stress', 'division', 'radius'],
                            output_fields=['secretion_rate', 'division'],
                            key=init_key,
                            transform_output={'division': lambda s,x: x*jax.nn.sigmoid(50*(s.radius - .45))*s.field},
                            expr_level_decay=1.,
                            interaction_init=jax.nn.initializers.constant(0.0)
                            ),     
    ])
    optimizer = optax.adam(1e-3)
    losses_t, models = [], []
    for i in range(N_OPT):
        opt_model = model
        key, train_key = jax.random.split(key)
        opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
        key, subkey = jax.random.split(key)
        rl, g = eqx.filter_value_and_grad(reinforce_loss)(opt_model, istate, cost_fn=COST_FN, key=subkey, n_steps=N_ADD, BATCH_SIZE=BATCH_SIZE, LAMBDA=LAMBDA)
        l = COST_FN(jxm.simulate(opt_model, istate, subkey, N_ADD, history=True)[0]).sum()
        losses = [float(l)]
        for i in range(EPOCHS):
            try:
                updates, opt_state = optimizer.update(g, opt_state, opt_model)
                opt_model = eqx.apply_updates(opt_model, updates)
                key, subkey = jax.random.split(key)

                rl, g = eqx.filter_value_and_grad(reinforce_loss)(opt_model, istate, cost_fn=COST_FN, key=subkey, n_steps=N_ADD, BATCH_SIZE=BATCH_SIZE, LAMBDA=LAMBDA)

                l = COST_FN(jxm.simulate(opt_model, istate, subkey, N_ADD, history=True)[0]).sum()
                losses.append(float(l))

            except FloatingPointError:
                print('NaN or Overflow')
                break
        losses_t.append(losses)
        models.append(opt_model)
    with open(ROOT_DIR + 'homoggrowth_' + str(N_OPT) + '_hid_' + str(N_HIDDEN) + '_chem_' + str(N_CHEM) + '_gene_loss', 'wb') as f:
        pickle.dump(losses_t, f)
    with open(ROOT_DIR + 'homoggrowth_' + str(N_OPT) + '_hid_' + str(N_HIDDEN) + '_chem_' + str(N_CHEM) + '_gene_models', 'wb') as f:
        eqx.tree_serialise_leaves(f, models)

if __name__ == '__main__':

    run_experiment()
