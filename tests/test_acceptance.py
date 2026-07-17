import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jax_morph as jxm
from jax_morph.control import GeneNetworkConnectionist, NeuralODE
from jax_morph.core.state import StateFieldSpec
from jax_morph.physics import (
    DIVISION_RATE,
    GROWTH_RATE,
    BrownianDynamics,
    Division,
    FreeScreenedDiffusion,
    MechanicalRelaxation,
    Morse,
    SaturatingCellGrowth,
    VirialStress,
)

N_DIM, N_TYPES = 2, 2

# These input specs must match the producing steps' write specs in name, shape, and heredity.
CHEMICAL = StateFieldSpec('chemical', shape=(1,), heritable=False)
STRESS = StateFieldSpec('stress', heritable=False)
SECRETION = StateFieldSpec('secretion_rate', shape=(1,))
_INPUTS = (CHEMICAL, STRESS)
_OUTPUTS = (GROWTH_RATE, DIVISION_RATE, SECRETION)


def build_model(key, decision):
    pot = Morse(epsilon=3.0, alpha=2.8)
    if decision == 'neural_ode':
        mlp = NeuralODE.make_mlp(_INPUTS, _OUTPUTS, hidden_size=4, key=key, width=16)
        dec = NeuralODE(_INPUTS, _OUTPUTS, hidden_size=4, mlp=mlp)
    else:
        kg, ki, kb = jax.random.split(key, 3)
        in_size, out_size, hidden = 2, 3, 4
        n_gene = hidden + out_size
        dec = GeneNetworkConnectionist(
            _INPUTS,
            _OUTPUTS,
            hidden_size=hidden,
            W_gene=0.1 * jax.random.normal(kg, (n_gene, n_gene)),
            W_in=0.1 * jax.random.normal(ki, (n_gene, in_size)),
            b=0.1 * jax.random.normal(kb, (n_gene,)),
        )

    # Cells sense fields at mechanical equilibrium before the controller advances.
    steps = [
        MechanicalRelaxation(pot, max_steps=200, f_tol=1e-3),
        FreeScreenedDiffusion(
            n_field_species=1,
            n_space_dim=N_DIM,
            diffusion=2.0,
            degradation=1.0,
        ),
        VirialStress(pot),
        dec,
        SaturatingCellGrowth(max_radius=0.7),
        Division(n_space_dim=N_DIM),
    ]
    return jxm.Model(steps)


@pytest.mark.acceptance
@pytest.mark.parametrize('decision', ['neural_ode', 'grn'])
def test_v1_acceptance(decision, key):
    model = build_model(key, decision)
    s0 = jxm.build_state_from_model(model).init_empty(
        capacity=48, n_space_dim=N_DIM, n_types=N_TYPES
    )
    # Seed live positions directly: update right-hand sides all read the pre-update alive mask.
    s0 = s0.update(
        alive=s0.alive.at[:3].set(True),
        radius=s0.radius.at[:3].set(0.3),
        celltype=s0.celltype.at[0, 0].set(1.0).at[1, 1].set(1.0).at[2, 0].set(1.0),
        position=s0.position.at[:3].set(0.5 * jax.random.normal(key, (3, N_DIM))),
        secretion_rate=jnp.ones((48, 1)),
        growth_rate=s0.growth_rate.at[:3].set(0.3),
        division_rate=s0.division_rate.at[:3].set(0.1),
    )

    final = jxm.simulate(model, s0, n_steps=12, dt=0.1, key=key)
    assert np.all(np.isfinite(np.asarray(final.position)))
    assert int(final.alive.sum()) >= 3

    def spread(m):
        traj = jxm.simulate(m, s0, n_steps=12, dt=0.1, key=key, history=True)
        final_frame = jax.tree_util.tree_map(lambda x: x[-1], traj)
        alive = final_frame.alive.astype(float)
        center = (final_frame.position * alive[:, None]).sum(0) / alive.sum()
        value = (((final_frame.position - center) ** 2).sum(1) * alive).sum()
        logp = jxm.trajectory_logp(m, traj, 0.1).sum()
        return value, logp

    def decision_grad_ok(gradient):
        leaves = [
            leaf for leaf in jax.tree_util.tree_leaves(gradient.steps[3]) if eqx.is_array(leaf)
        ]
        finite = all(np.all(np.isfinite(np.asarray(leaf))) for leaf in leaves)
        nonzero = any(np.linalg.norm(np.asarray(leaf)) > 1e-8 for leaf in leaves)
        return bool(leaves) and finite and nonzero

    g_path = eqx.filter_grad(lambda m: spread(m)[0])(model)
    path_leaves = [leaf for leaf in jax.tree_util.tree_leaves(g_path) if eqx.is_array(leaf)]
    assert path_leaves and all(np.all(np.isfinite(np.asarray(leaf))) for leaf in path_leaves)
    assert decision_grad_ok(g_path)

    def reinforce_loss(m):
        value, logp = spread(m)
        return -logp * jax.lax.stop_gradient(value)

    g_reinforce = eqx.filter_grad(reinforce_loss)(model)
    reinforce_leaves = [
        leaf for leaf in jax.tree_util.tree_leaves(g_reinforce) if eqx.is_array(leaf)
    ]
    assert reinforce_leaves and all(
        np.all(np.isfinite(np.asarray(leaf))) for leaf in reinforce_leaves
    )
    assert decision_grad_ok(g_reinforce)


def test_pathwise_gradient_matches_finite_difference():
    def separation(rate):
        model = jxm.Model(
            [
                MechanicalRelaxation(Morse(), max_steps=400, f_tol=1e-8),
                SaturatingCellGrowth(max_radius=2.0),
            ]
        )
        s0 = jxm.build_state_from_model(model).init_empty(capacity=4, n_space_dim=2, n_types=1)
        s0 = s0.update(
            alive=s0.alive.at[:2].set(True),
            radius=s0.radius.at[:2].set(0.5),
            position=jnp.array([[0.0, 0.0], [0.9, 0.0], [0.0, 0.0], [0.0, 0.0]]),
            growth_rate=s0.growth_rate.at[:2].set(rate),
        )
        final = jxm.simulate(model, s0, n_steps=3, dt=0.1)
        return jnp.linalg.norm(final.position[0] - final.position[1])

    g_auto = float(jax.grad(separation)(jnp.array(0.5)))
    g_fd = (float(separation(jnp.array(0.503))) - float(separation(jnp.array(0.497)))) / 0.006
    assert np.isclose(g_auto, g_fd, rtol=2e-2, atol=1e-3)


def test_forward_euler_dt_convergence():
    # This isolates one genuinely O(dt) dynamic step rather than the macro-step's multi-phase
    # splitting. SaturatingCellGrowth is exact in dt, so it cannot expose Euler convergence.
    # BrownianDynamics at kT=0 supplies deterministic forward-Euler drift instead. High drag and a
    # start just outside the Morse well keep the pair in a well-resolved, mid-transit regime at T=1:
    # it neither overshoots nor reaches the flat fixed point, where errors would collapse to machine
    # precision. Every resolution is compared with one stable high-resolution reference rather than
    # with its adjacent resolution, so error halving measures convergence to a common trajectory.
    def final_separation(n_steps):
        model = jxm.Model([BrownianDynamics(Morse(), n_space_dim=2, gamma=20.0, kT=0.0)])
        s0 = jxm.build_state_from_model(model).init_empty(capacity=2, n_space_dim=2, n_types=1)
        s0 = s0.update(
            alive=s0.alive.at[:2].set(True),
            radius=s0.radius.at[:2].set(0.5),
            position=jnp.array([[0.0, 0.0], [1.3, 0.0]]),
        )
        final = jxm.simulate(
            model,
            s0,
            n_steps=n_steps,
            dt=1.0 / n_steps,
            key=jax.random.PRNGKey(0),
        )
        return float(jnp.linalg.norm(final.position[0] - final.position[1]))

    reference = final_separation(5120)
    errors = [abs(final_separation(n_steps) - reference) for n_steps in (20, 40, 80)]
    ratios = [errors[index] / errors[index + 1] for index in range(len(errors) - 1)]
    assert all(ratio > 1.7 for ratio in ratios)


class ConstDivisionRate(jxm.SimulationStep):
    # Store the scalar division rate on the model and rewrite it into live state during replay. The
    # initial state is a detached conditioning boundary, so seeding this rate only in state would not
    # test whether the score-function gradient reaches a model parameter.
    step_type = jxm.StepType.QUASISTATIC
    rate: jax.Array

    def state_writes(self):
        return (DIVISION_RATE,)

    def __call__(self, state, *, dt, key):
        del dt, key
        return state.set('division_rate', jnp.where(state.alive, self.rate, 0.0))


def test_reinforce_gradient_is_unbiased(key):
    # One cell divides with p = 1 - exp(-lambda * dt), and the reward is the daughter count. The
    # Monte Carlo mean of R * d(logp)/d(lambda) should therefore match
    # dE[R]/d(lambda) = dt * exp(-lambda * dt).
    lam0, dt = 0.7, 1.0
    model = jxm.Model([ConstDivisionRate(jnp.array(lam0)), Division(n_space_dim=2)])
    s0 = jxm.build_state_from_model(model).init_empty(capacity=2, n_space_dim=2, n_types=1)
    s0 = s0.update(alive=s0.alive.at[0].set(True), radius=s0.radius.at[0].set(0.5))

    def estimator(sample_key):
        next_state = model(s0, dt=dt, key=sample_key)
        reward = jnp.sum(next_state.alive.astype(jnp.float64)) - 1.0
        gradient = eqx.filter_grad(
            lambda candidate: jxm.transition_logp(candidate, s0, next_state, dt)
        )(model)
        return reward * gradient.steps[0].rate

    estimate = float(jnp.mean(jax.vmap(estimator)(jax.random.split(key, 16384))))
    analytic = dt * np.exp(-lam0 * dt)
    assert np.isclose(estimate, analytic, atol=0.03)
