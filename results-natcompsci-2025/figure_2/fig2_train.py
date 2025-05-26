import os
import time

import pickle

# JAX Imports
import jax
import jax.numpy as np
import matplotlib.pyplot as plt

# JAX-Morph Imports
import jax_morph as jxm  # type: ignore

import equinox as eqx

from fig2_istate_and_model import build_istate, build_model

jax.config.update("jax_debug_nans", True)
jax.config.update("jax_enable_x64", True)
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".95"

# Change working directory to script location
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)


def plot_reward(log, run_id):

    plt.plot(log.rewards, "r")
    plt.xlabel("Epoch")
    plt.ylabel("Reward")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(run_id, f"reward_{run_id}.png"))
    plt.close()


def plot_spheres_div(model, istate, subkey, run_id, trained=False):

    trained = "trained" if trained else "init"

    N_ADD = int(istate.celltype.shape[0] - istate.celltype.sum(-1).sum(-1))

    fstate_opt, _ = jxm.simulate(model, istate, key=subkey, n_steps=N_ADD)

    fig, ax = jxm.visualization.draw_spheres_division(
        fstate_opt,
        grid=True,
        elev=0,
        azim=0,
    )

    # axis labels
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    plt.tight_layout()

    plt.savefig(os.path.join(run_id, f"spheres_div_{trained}_{run_id}.png"))
    plt.close()


if __name__ == "__main__":

    # Define rewards function (maximize sum of squares along y-axis)
    rewards_fn = jxm.opt.reward_ssq_diff(coordinate_idx=1, center_of_mass=True)

    # create Reinforce loss function
    rloss_fn = jxm.opt.reinforce_loss()

    # add L1 regularization to interaction matrix
    loss_fn = lambda model, trajectory, rewards, sim_keys: rloss_fn(
        model, trajectory, rewards, sim_keys
    ) + 1e0 * np.mean(np.abs(model.GeneNetwork.interaction_matrix))

    # BUILD INITIAL STATE
    key = jxm.utils.generate_random_key()

    istate = build_istate(
        N_HIDDEN_GENES=1,
        N_INIT=15,
        N_TOT=150,
        N_DIM=3,
    )

    # BUILD INITIAL MODEL
    key, init_key = jax.random.split(key)
    model = build_model(init_key, istate)

    # TRAIN MODEL
    key, train_key = jax.random.split(key)

    trained_model, log = jxm.opt.train_reinforce(
        train_key,
        model,
        istate,
        rewards_fn,
        loss_fn,
        epochs=1000,
        learning_rate=5e-3,
        batch_size=4,
        return_discount=0.6,
    )

    # Create folder for this training run
    run_id = f"run_{int(time.time())}"
    os.makedirs(run_id, exist_ok=True)

    # Save training log and model with error handling
    try:
        log_path = os.path.join(run_id, f"training_log_{run_id}.pkl")
        init_model_path = os.path.join(run_id, f"init_model_{run_id}.eqx")
        trained_model_path = os.path.join(run_id, f"trained_model_{run_id}.eqx")

        # Create modified log with None values for large attributes
        log = log._replace(
            static_model=None,
            saved_models=None,
            istate=None,
        )

        # Save with highest protocol for better compatibility
        with open(log_path, "wb") as f:
            pickle.dump(log, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"Successfully saved log to {run_id}/")

        eqx.tree_serialise_leaves(init_model_path, model)
        print(f"Successfully saved init model to {run_id}/")

        eqx.tree_serialise_leaves(trained_model_path, trained_model)
        print(f"Successfully saved trained model to {run_id}/")

        # diagnostic plots
        print("Plotting reward...")
        plot_reward(log, run_id)
        print("Plotting initial spheres...")
        plot_spheres_div(model, istate, key, run_id, trained=False)
        print("Plotting trained spheres...")
        plot_spheres_div(trained_model, istate, key, run_id, trained=True)
        print("Successfully saved plots")

    except Exception as e:
        print(f"Error saving files: {str(e)}")
