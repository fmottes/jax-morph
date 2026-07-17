"""Tests for the VirialStress sensing step.

VirialStress exposes ``PairwisePotential.virial_pressure`` as a per-cell ``stress`` field. The
pressure itself is covered in test_mechanics; here we check the step wiring: it writes ``stress``,
respects the sign/cutoff, masks dead cells, composes in a Model, and stays differentiable through a
traced potential parameter.
"""

import jax
import jax.numpy as jnp
import numpy as np

import jax_morph as jxm
from jax_morph.core.state import build_state_from_model
from jax_morph.physics.mechanics import Morse, VirialStress


def _pair(dist, radius=0.5):
    # Build the state from a VirialStress step so the schema carries the `stress` field it writes.
    step = VirialStress(Morse())
    s = build_state_from_model(step).init_empty(capacity=2, n_space_dim=2, n_types=1)
    return s.update(
        alive=s.alive.at[:2].set(True),
        radius=s.radius.at[:2].set(radius),
        position=jnp.array([[0.0, 0.0], [dist, 0.0]]),
        celltype=s.celltype.at[:2, 0].set(1.0),
    )


def test_compressed_pair_has_positive_stress_and_far_pair_zero():
    vs = VirialStress(Morse())
    compressed = vs(_pair(0.6), dt=1.0, key=None)['stress']  # r < sigma=1.0 -> repulsive -> p > 0
    apart = vs(_pair(3.0), dt=1.0, key=None)['stress']  # beyond cutoff -> ~0
    assert float(compressed[0]) > 0.0 and float(compressed[1]) > 0.0
    assert np.allclose(np.asarray(apart), 0.0, atol=1e-6)


def test_stress_matches_potential_virial_pressure():
    # The step is a thin wrapper: the written field equals potential.virial_pressure exactly.
    pot = Morse()
    s = _pair(0.7)
    out = VirialStress(pot)(s, dt=1.0, key=None)
    assert np.allclose(np.asarray(out['stress']), np.asarray(pot.virial_pressure(s)))


def test_stress_is_zero_on_dead_cells():
    vs = VirialStress(Morse())
    s = _pair(0.6)
    s = s.set('alive', s.alive.at[1].set(False))  # kill the partner
    out = vs(s, dt=1.0, key=None)['stress']
    assert float(out[1]) == 0.0  # dead cell scores 0
    assert np.isclose(float(out[0]), 0.0)  # its only neighbour is dead -> no interaction -> 0


def test_stress_step_composes_in_a_model_and_advances_time(key):
    # As a quasistatic step it advances one macro-step, writing stress into the state.
    model = jxm.Model([VirialStress(Morse())])
    s = _pair(0.6)
    s1 = model(s, dt=1.0, key=key)
    assert float(s1.stress[0]) > 0.0 and np.isclose(float(s1.t), 1.0)


def test_stress_is_differentiable_through_a_traced_potential_param():
    # A traced potential parameter reaches the written stress, so a stress-based loss is optimizable.
    s = _pair(0.6)

    def loss(epsilon):
        step = VirialStress(Morse(epsilon=epsilon))
        out = step(s, dt=1.0, key=None)
        return jnp.sum(out['stress'])

    g = jax.grad(loss)(jnp.array(3.0))
    assert np.isfinite(float(g)) and float(g) != 0.0
