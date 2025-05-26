import os
import json

# JAX Imports
import jax
import optax

# JAX-Morph Imports
import jax_morph as jxm  # type: ignore

import equinox as eqx

from fig3_istate_and_model import build_istate, build_model

jax.config.update("jax_debug_nans", True)
jax.config.update("jax_enable_x64", True)
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".95"

# Change working directory to script location
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)


if __name__ == "__main__":

    N_CHEM = 2

    key = jxm.utils.generate_random_key()
    istate = build_istate(key)

    # Optimization parameters
    N_ADD = int(istate.celltype.shape[0] - istate.celltype.sum(-1).sum(-1))

    N_OPT_RUNS = 15

    EPOCHS = 300
    N_EPISODES = 8
    N_VAL_EPISODES = 8

    GAMMA = 0.9
    LAMBDA = 10.0

    COST_FN = jxm.opt._old.cost_functions.CellTypeImbalance(metric="number")

    LOSS = jxm.opt._old.losses.ReinforceLoss(
        COST_FN,
        n_sim_steps=N_ADD,
        n_episodes=N_EPISODES,
        n_val_episodes=N_VAL_EPISODES,
        gamma=GAMMA,
        lambda_l1=LAMBDA,
        normalize_cost_returns="batch",
    )

    OPTIMIZER = optax.adam(1e-2)

    root_dir = "./new_trained_models/"

    # dump json with opt hyperparams
    with open(root_dir + "train-ChemHomeo-opt-hyperparams.json", "w") as f:
        json.dump(
            {
                "N_OPT_RUNS": N_OPT_RUNS,
                "EPOCHS": EPOCHS,
                "N_EPISODES": N_EPISODES,
                "N_VAL_EPISODES": N_VAL_EPISODES,
                "GAMMA": GAMMA,
                "LAMBDA": LAMBDA,
                "N_CHEM": N_CHEM,
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

        os.makedirs(root_dir + f"train-ChemHomeo-{i}", exist_ok=True)

        eqx.tree_serialise_leaves(
            root_dir + f"train-ChemHomeo-{i}/init-ChemHomeo-{i}.eqx",
            model,
        )
        eqx.tree_serialise_leaves(
            root_dir + f"train-ChemHomeo-{i}/trained-ChemHomeo-{i}.eqx",
            opt_model,
        )
        eqx.tree_serialise_leaves(
            root_dir + f"train-ChemHomeo-{i}/results-ChemHomeo-{i}.eqx",
            opt_results,
        )
