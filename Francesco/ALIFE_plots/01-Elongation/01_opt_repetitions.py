
# LIMIT GPU MEMORY TAKEN UP BY THE NOTEBOOK
# you can specify the fraction of the AVAILABLE GPU memory that will be
# pre-allocated (jax default is .9)

import os
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.95'
# #os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

# #use another GPU if the default one is occupied
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

#import utils from the parent folder
import sys
sys.path.append('../')

#no warnings about floating precision
import warnings
warnings.filterwarnings('ignore')

import pickle

########## IMPORT JAX ECOSYSTEM ##########
import jax
import jax.numpy as np
from jax.random import split, PRNGKey

jax.config.update('jax_debug_nans', True)


########## IMPORT OTHER UTILITIES ##########
############################################
from collections import namedtuple

from alife_utils import default_params, build_sim_from_params, train, run_optimization



########## DEFINE METRIC ##########
def position_sum_of_squares(state):

    alive = state.celltype > 0
    n = np.sum(alive)

    m = np.sum((state.position[:, 1] * alive)**2)/n 

    return m



def run_experiment():

    key = PRNGKey(0)


    key, init_key = split(key)
    Dparams, Dtrain_params = default_params(init_key)

    key, subkey = split(key)
    sim = build_sim_from_params(Dparams, Dtrain_params, subkey)


    LogExperiment = namedtuple('LogExperiment', ['epochs', 'episodes_per_update', 'episodes_per_eval', 'learning_rate', 'save_every', 'sim', 'opt_runs'])

    LogRep = namedtuple('LogRep', ['loss_t', 'params_t', 'grads_t'])


    N_OPT = 30

    EPOCHS = 300
    EPISODES_PER_UPDATE = 4
    EPISODES_PER_EVAL = 100

    LEARNING_RATE = 1e-3

    METRIC_FN = position_sum_of_squares


    SAVE_EVERY = 10
    log = LogExperiment(epochs=EPOCHS, 
                    episodes_per_update=EPISODES_PER_UPDATE, 
                    episodes_per_eval=EPISODES_PER_EVAL, 
                    learning_rate=LEARNING_RATE, 
                    save_every=SAVE_EVERY,
                    sim=sim, 
                    opt_runs=[]
                    )


    for i in range(N_OPT):
        key, train_key = split(key)
        loss_t, params_t, _ = run_optimization(train_key,
                                            sim,
                                            METRIC_FN,
                                            metric_type='cost',
                                            epochs=EPOCHS,
                                            episodes_per_update=EPISODES_PER_UPDATE,
                                            episodes_per_eval=EPISODES_PER_EVAL,
                                            learning_rate=LEARNING_RATE,
                                            save_every=SAVE_EVERY
                                            )

        log.opt_runs.append(LogRep(loss_t=loss_t, params_t=params_t, grads_t=None))


    with open('log_opt_reps.pkl', 'wb') as f:
        pickle.dump(log, f)

    return


if __name__ == '__main__':
    run_experiment()