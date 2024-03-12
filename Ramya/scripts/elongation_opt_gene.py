
# LIMIT GPU MEMORY TAKEN UP BY THE NOTEBOOK
# you can specify the fraction of the AVAILABLE GPU memory that will be
# pre-allocated (jax default is .9)

import os
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.95'

#import utils from the parent folder
import sys
sys.path.append("../")
sys.path.append('/n/home10/rdeshpande/morphogenesis/jax-morph')
ROOT_DIR = '/n/home10/rdeshpande/morphogenesis/data/paper/'

#no warnings about floating precision
import warnings
warnings.filterwarnings('ignore')

import pickle
from optax import piecewise_constant_schedule

########## IMPORT JAX ECOSYSTEM ##########
import jax
import jax.numpy as np
from jax.random import split, PRNGKey

jax.config.update('jax_debug_nans', True)


########## IMPORT OTHER UTILITIES ##########
############################################
from collections import namedtuple
from Ramya.utils import gene_utils



def run_experiment():

    key = PRNGKey(0)
    init_key, subkey = split(PRNGKey(0), 2)
    params_gn, train_params_gn = gene_utils.default_params(init_key, n_chem=3)
    use_state_fields = gene_utils.CellState(position=False, celltype=False, radius=True, chemical=True,chemgrad=True,field=False,divrate=True,gene_vec=False,stress=True,key=False)
    n_inputs = params_gn["n_chem"] + 2*params_gn["n_chem"] + 3
    sim_gn = gene_utils.build_sim_from_params(params_gn, train_params_gn, use_state_fields, subkey, n_inputs=n_inputs)


    LogExperiment = namedtuple('LogExperiment', ['epochs', 'episodes_per_update', 'episodes_per_eval', 'learning_rate', 'save_every', 'sim', 'opt_runs'])

    LogRep = namedtuple('LogRep', ['loss_t', 'params_t', 'grads_t'])


    N_OPT = int(sys.argv[1]) #100

    EPOCHS = int(sys.argv[2]) #400
    EPISODES_PER_UPDATE = int(sys.argv[3]) #4
    EPISODES_PER_EVAL = int(sys.argv[4]) #100

    LEARNING_RATE = 1e-3

    METRIC_FN = gene_utils.position_sum_of_squares


    SAVE_EVERY = 10
    log = LogExperiment(epochs=EPOCHS, 
                    episodes_per_update=EPISODES_PER_UPDATE, 
                    episodes_per_eval=EPISODES_PER_EVAL, 
                    learning_rate=LEARNING_RATE, 
                    save_every=SAVE_EVERY,
                    sim=sim_gn, 
                    opt_runs=[]
                    )

    loss_tt = []
    params_tt = []

    for i in range(N_OPT):
        key, train_key = split(key)
        loss_t, params_t, _ = gene_utils.run_optimization(train_key,
                                                sim_gn,
                                                METRIC_FN,
                                                metric_type='cost',
                                                epochs=EPOCHS,
                                                episodes_per_update=EPISODES_PER_UPDATE,
                                                episodes_per_eval=EPISODES_PER_EVAL,
                                                learning_rate=LEARNING_RATE,
                                                save_every=SAVE_EVERY
                                                )
        log.opt_runs.append(LogRep(loss_t=loss_t, params_t=params_t, grads_t=None))
        loss_tt.append(loss_t)
        params_tt.append(params_t)


    with open(ROOT_DIR + 'elongation_' + str(N_OPT) + '_gene_loss.pkl', 'wb') as f:
        pickle.dump(loss_tt, f)
    with open(ROOT_DIR + 'elongation_' + str(N_OPT) + '_gene_params.pkl', 'wb') as f:
        pickle.dump(params_tt, f)
    return


if __name__ == '__main__':

    run_experiment()
