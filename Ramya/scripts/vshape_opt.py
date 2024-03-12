
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
from Francesco.ALIFE_plots.alife_utils import default_params, build_sim_from_params, run_optimization, mask_metric



########## DEFINE METRIC ##########
def v_mask(pos):
    '''
    Constrain cell growth in a V shape.
    '''
    return (pos[:,1]+1.5 > .5*np.abs(pos[:,0])) * (pos[:,1]+1.5 < 3.5+.5*np.abs(pos[:,0])) * (pos[:,1]>0.)


v_metric = mask_metric(v_mask)



def run_experiment():

    key = PRNGKey(0)

    key, init_key = split(key)
    Dparams, Dtrain_params = default_params(init_key, 3)

    key, subkey = split(key)
    sim = build_sim_from_params(Dparams, Dtrain_params, subkey)


    LogExperiment = namedtuple('LogExperiment', ['epochs', 'episodes_per_update', 'episodes_per_eval', 'learning_rate', 'save_every', 'sim', 'opt_runs'])

    LogRep = namedtuple('LogRep', ['loss_t', 'params_t', 'grads_t'])


    N_OPT = int(sys.argv[1]) #100

    EPOCHS = int(sys.argv[2]) #400
    EPISODES_PER_UPDATE = int(sys.argv[3]) #4
    EPISODES_PER_EVAL = int(sys.argv[4]) #100

    LEARNING_RATE = piecewise_constant_schedule(2e-3, {100: .1})

    METRIC_FN = v_metric


    SAVE_EVERY = 10
    log = LogExperiment(epochs=EPOCHS, 
                    episodes_per_update=EPISODES_PER_UPDATE, 
                    episodes_per_eval=EPISODES_PER_EVAL, 
                    learning_rate=LEARNING_RATE, 
                    save_every=SAVE_EVERY,
                    sim=sim, 
                    opt_runs=[]
                    )

    loss_tt = []
    params_tt = []

    for i in range(N_OPT):
        key, train_key = split(key)
        loss_t, params_t, _ = run_optimization(train_key,
                                                sim,
                                                METRIC_FN,
                                                metric_type='reward',
                                                epochs=EPOCHS,
                                                episodes_per_update=EPISODES_PER_UPDATE,
                                                episodes_per_eval=EPISODES_PER_EVAL,
                                                learning_rate=LEARNING_RATE,
                                                save_every=SAVE_EVERY
                                                )
        log.opt_runs.append(LogRep(loss_t=loss_t, params_t=params_t, grads_t=None))
        loss_tt.append(loss_t)
        params_tt.append(params_t)


    with open(ROOT_DIR + 'vshape_' + str(N_OPT) + '_loss.pkl', 'wb') as f:
        pickle.dump(loss_tt, f)
    with open(ROOT_DIR + 'vshape_' + str(N_OPT) + '_params.pkl', 'wb') as f:
        pickle.dump(params_tt, f)
    return


if __name__ == '__main__':

    run_experiment()