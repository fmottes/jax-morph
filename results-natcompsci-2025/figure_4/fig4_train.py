import os
import json

import shutil

# JAX Imports
import jax
import optax

# JAX-Morph Imports
import jax_morph as jxm  # type: ignore

import equinox as eqx

# Local Imports
from fig4_istate_and_model import build_istate, build_model

jax.config.update("jax_debug_nans", True)
jax.config.update("jax_enable_x64", True)
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".95"


# Change working directory to script location
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

if __name__ == "__main__":

    key = jxm.utils.generate_random_key()
    key, init_key = jax.random.split(key)

    istate = build_istate(init_key)

    # Optimization parameters
    N_ADD = int(istate.celltype.shape[0] - istate.celltype.sum(-1).sum(-1))

    N_OPT_RUNS = 10
    EPOCHS = 300
    N_EPISODES = 4
    N_VAL_EPISODES = 16

    LAMBDA = 0.1

    COST_FN = jxm.opt._old.cost_functions.CVDivrates()
    LOSS = jxm.opt._old.losses.SimpleLoss(
        COST_FN,
        n_sim_steps=N_ADD,
        n_episodes=N_EPISODES,
        n_val_episodes=N_VAL_EPISODES,
        lambda_l1=LAMBDA,
        normalize_cost_returns=False,
    )

    OPTIMIZER = optax.adam(1e-3)

    root_dir = "./trained_models/"

    # !!! empty training_runs folder
    try:
        shutil.rmtree(root_dir)
    except FileNotFoundError:
        pass

    os.makedirs(root_dir, exist_ok=True)

    # dump json with opt hyperparams
    with open(root_dir + "train-HomGrowth-opt-hyperparams.json", "w") as f:
        json.dump(
            {
                "N_OPT_RUNS": N_OPT_RUNS,
                "EPOCHS": EPOCHS,
                "N_EPISODES": N_EPISODES,
                "N_VAL_EPISODES": N_VAL_EPISODES,
                "LAMBDA": LAMBDA,
            },
            f,
        )

    for i in range(N_OPT_RUNS):

        key, init_key, train_key = jax.random.split(key, 3)

        model = build_model(init_key, istate)

        opt_model, opt_results = jxm.opt._old.training.train(
            model,
            istate,
            LOSS,
            key=train_key,
            epochs=EPOCHS,
            optimizer=OPTIMIZER,
            model_save_every=None,
            grad_save_every=None,
        )

        os.makedirs(root_dir + f"train-HomGrowth-{i}", exist_ok=True)

        eqx.tree_serialise_leaves(
            root_dir + f"train-HomGrowth-{i}/init-HomGrowth-{i}.eqx", model
        )
        eqx.tree_serialise_leaves(
            root_dir + f"train-HomGrowth-{i}/trained-HomGrowth-{i}.eqx",
            opt_model,
        )
        eqx.tree_serialise_leaves(
            root_dir + f"train-HomGrowth-{i}/results-HomGrowth-{i}.eqx",
            opt_results,
        )
