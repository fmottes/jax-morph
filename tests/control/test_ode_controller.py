import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jax_morph as jxm
from jax_morph.control.ode import (
    GeneNetworkConnectionist,
    GeneNetworkMWC,
    NeuralODE,
    ODEController,
    _rescaled_sigmoid,
)
from jax_morph.core.state import StateFieldSpec, build_state_from_model
from jax_morph.physics import GROWTH_RATE, SaturatingCellGrowth

_IN = (StateFieldSpec('chemical', shape=(1,), heritable=False),)
_OUT = (StateFieldSpec('secretion_rate', shape=(1,)),)


def test_ode_controller_base_is_abstract():
    with pytest.raises(TypeError):
        ODEController(_IN, _OUT, hidden_size=3, tag='x')


def test_gene_network_delta_shapes_and_finite(key):
    gn = GeneNetworkConnectionist(_IN, _OUT, hidden_size=3)
    s = build_state_from_model(gn).init_empty(capacity=2, n_space_dim=2, n_types=1)
    s = s.set('alive', s.alive.at[0].set(True))
    d = gn(s, dt=0.1, key=key)
    assert d.gene_hidden.shape == (2, 3)
    assert d.secretion_rate.shape == (2, 1)
    assert np.all(np.isfinite(np.asarray(d.gene_hidden)))


def test_explicit_parameters_are_used():
    # the constructor takes the regulatory weights directly (no random init): with a zero interaction
    # matrix and no degradation, the derivative is exactly the sigmoid of the supplied bias.
    outputs = (StateFieldSpec('genes', shape=(2,)),)
    b = jnp.array([-2.0, 3.0])
    gn = GeneNetworkConnectionist((), outputs, hidden_size=0, b=b, gamma=0.0)
    derivative = gn.vector_field(0.0, jnp.zeros((1, 2)), jnp.zeros((1, 0)))
    assert np.allclose(np.asarray(derivative), np.asarray(_rescaled_sigmoid(b))[None, :])


def test_list_matrix_stays_static_leaf():
    # a matrix passed as a Python list is stored verbatim, so it is a non-array (static, frozen) leaf
    # that training skips; jnp.asarray at point of use lets it still drive the vector field.
    outputs = (StateFieldSpec('genes', shape=(1,)),)
    gn = GeneNetworkConnectionist((), outputs, hidden_size=0, W_gene=[[0.5]])
    assert gn.W_gene == [[0.5]]
    assert not any(eqx.is_array(leaf) for leaf in jax.tree_util.tree_leaves(gn.W_gene))
    derivative = gn.vector_field(0.0, jnp.array([[1.0]]), jnp.zeros((1, 0)))
    assert np.all(np.isfinite(np.asarray(derivative)))


def test_parameter_shape_validation():
    outputs = (StateFieldSpec('genes', shape=(2,)),)  # n_gene = 2
    with pytest.raises(ValueError):
        GeneNetworkConnectionist((), outputs, hidden_size=0, W_gene=jnp.zeros((3, 3)))


def test_connectionist_sigmoid_saturates_for_large_finite_drive():
    outputs = (StateFieldSpec('genes', shape=(2,)),)
    gn = GeneNetworkConnectionist((), outputs, hidden_size=0, b=jnp.array([-1e200, 1e200]))
    derivative = gn.vector_field(0.0, jnp.zeros((1, 2)), jnp.zeros((1, 0)))
    assert np.all(np.isfinite(np.asarray(derivative)))
    assert np.allclose(np.asarray(derivative), [[0.0, 1.0]])


def test_hidden_size_zero_allocates_no_hidden_field(key):
    gn = GeneNetworkConnectionist(_IN, _OUT, hidden_size=0)
    assert all('hidden' not in spec.name for spec in gn.state_writes())
    s = build_state_from_model(gn).init_empty(capacity=2, n_space_dim=2, n_types=1)
    s = s.set('alive', s.alive.at[0].set(True))
    d = gn(s, dt=0.1, key=key)
    assert d.secretion_rate.shape == (2, 1)
    assert np.all(np.isfinite(np.asarray(d.secretion_rate)))


def test_pathwise_gradient_reaches_controller_params(key):
    # bias the drive so the outputs move, then check grad of a scalar of the delta wrt the weights is
    # finite and nonzero (it flows through the diffrax adjoint) - the pathwise control-layer estimator.
    gn = GeneNetworkConnectionist(_IN, _OUT, hidden_size=3, b=jnp.ones((4,)))
    s = build_state_from_model(gn).init_empty(capacity=2, n_space_dim=2, n_types=1)
    s = s.set('alive', s.alive.at[0].set(True))
    grads = eqx.filter_grad(lambda m: jnp.sum(m(s, dt=0.1, key=key).secretion_rate))(gn)
    leaves = [g for g in jax.tree_util.tree_leaves(grads) if eqx.is_array(g)]
    assert leaves and all(np.all(np.isfinite(np.asarray(g))) for g in leaves)
    assert sum(float(np.abs(np.asarray(g)).sum()) for g in leaves) > 0.0


def test_neural_ode_delta_shapes_and_finite(key):
    mlp = NeuralODE.make_mlp(_IN, _OUT, hidden_size=3, key=key, width=8, depth=1)
    node = NeuralODE(_IN, _OUT, hidden_size=3, mlp=mlp)
    s = build_state_from_model(node).init_empty(capacity=2, n_space_dim=2, n_types=1)
    s = s.set('alive', s.alive.at[0].set(True))
    d = node(s, dt=0.1, key=key)
    assert d.ode_hidden.shape == (2, 3)
    assert d.secretion_rate.shape == (2, 1)
    assert np.all(np.isfinite(np.asarray(d.ode_hidden)))


def test_neural_ode_make_mlp_and_size_validation(key):
    mlp = NeuralODE.make_mlp(_IN, _OUT, hidden_size=2, key=key, width=8, depth=1)
    node = NeuralODE(_IN, _OUT, hidden_size=2, mlp=mlp)
    assert node.mlp.in_size == 1 + 3  # in_size 1 + evolving (hidden 2 + out 1)
    assert node.mlp.out_size == 3
    with pytest.raises(ValueError):
        NeuralODE(_IN, _OUT, hidden_size=5, mlp=mlp)  # mlp sized for hidden_size=2, not 5


def test_gene_network_mwc_delta_shapes_and_finite(key):
    drivers = (StateFieldSpec('maternal', shape=(3,)),)
    genes = (StateFieldSpec('gap', shape=(2,)),)
    gn = GeneNetworkMWC(drivers, genes)
    assert all('hidden' not in spec.name for spec in gn.state_writes())
    s = build_state_from_model(gn).init_empty(capacity=2, n_space_dim=2, n_types=1)
    s = s.set('alive', s.alive.at[0].set(True))
    d = gn(s, dt=0.1, key=key)
    assert d.gap.shape == (2, 2)
    assert np.all(np.isfinite(np.asarray(d.gap)))


def test_gene_network_mwc_supports_latent_genes(key):
    drivers = (StateFieldSpec('maternal', shape=(3,)),)
    genes = (StateFieldSpec('gap', shape=(2,)),)
    gn = GeneNetworkMWC(drivers, genes, hidden_size=4)
    assert jnp.asarray(gn.H_gene).shape == (6, 6)
    s = build_state_from_model(gn).init_empty(capacity=2, n_space_dim=2, n_types=1)
    s = s.set('alive', s.alive.at[0].set(True))
    d = gn(s, dt=0.1, key=key)
    assert d.gene_hidden.shape == (2, 4)
    assert d.gap.shape == (2, 2)
    assert np.all(np.isfinite(np.asarray(d.gene_hidden)))


def test_gene_network_mwc_log_parameters_remain_finite_at_dtype_limits():
    drivers = (StateFieldSpec('maternal', shape=(1,)),)
    genes = (StateFieldSpec('gap', shape=(1,)),)
    gn = GeneNetworkMWC(
        drivers,
        genes,
        log_rho=jnp.full((1,), 1000.0),
        log_tau=jnp.full((1,), -1000.0),
        log_K_gene=jnp.full((1, 1), -1000.0),
        log_K_in=jnp.full((1, 1), -1000.0),
    )
    derivative = gn.vector_field(0.0, jnp.ones((1, 1)), jnp.ones((1, 1)))
    assert np.all(np.isfinite(np.asarray(derivative)))


def test_gene_network_mwc_finite_with_tiny_thresholds_and_large_genes():
    # tiny thresholds (K -> dtype min) with large genes overflow g/K to +inf; a mixed-sign
    # interaction row then risks (+inf) + (-inf) = NaN in the occupancy sum. The clamped ratio must
    # keep the vector field finite under the always-on jax_debug_nans.
    genes = (StateFieldSpec('gap', shape=(2,)),)
    gn = GeneNetworkMWC(
        (),
        genes,
        log_K_gene=jnp.full((2, 2), -1000.0),
        H_gene=jnp.array([[1.0, -1.0], [-1.0, 1.0]]),
    )
    derivative = gn.vector_field(0.0, jnp.full((1, 2), 1e6), jnp.zeros((1, 0)))
    assert np.all(np.isfinite(np.asarray(derivative)))


def test_controller_composes_in_model_and_masks_dead_cells(key):
    # a controller output field feeds a downstream physics step, and the model alive-masks the
    # controller's dynamic delta so dead cells are never integrated.
    gn = GeneNetworkConnectionist(_IN, (GROWTH_RATE,), hidden_size=1, b=jnp.ones((2,)))
    model = jxm.Model([gn, SaturatingCellGrowth(max_radius=1.0)])
    s = build_state_from_model(model).init_empty(capacity=3, n_space_dim=2, n_types=1)
    s = s.update(alive=s.alive.at[:2].set(True), radius=s.radius.at[:2].set(0.3))
    out = model(s, dt=0.5, key=key)
    assert np.all(np.isfinite(np.asarray(out.growth_rate)))
    assert not np.isclose(float(out.growth_rate[0]), float(s.growth_rate[0]))  # alive cell updated
    assert float(out.growth_rate[2]) == float(s.growth_rate[2])  # dead cell masked (unchanged)


def test_controller_multi_output_and_scalar_field_split(key):
    # the delta-splitting loop handles several output fields, including a scalar (shape ()) one.
    outputs = (StateFieldSpec('scalar_out'), StateFieldSpec('vector_out', shape=(2,)))
    gn = GeneNetworkConnectionist(_IN, outputs, hidden_size=1)
    s = build_state_from_model(gn).init_empty(capacity=2, n_space_dim=2, n_types=1)
    s = s.set('alive', s.alive.at[0].set(True))
    d = gn(s, dt=0.1, key=key)
    assert d.scalar_out.shape == (2,)  # scalar field -> per-cell scalar (chunk[:, 0])
    assert d.vector_out.shape == (2, 2)  # vector field -> reshaped
    assert np.all(np.isfinite(np.asarray(d.scalar_out)))
    assert np.all(np.isfinite(np.asarray(d.vector_out)))


def test_controller_with_no_drivers(key):
    # input_specs=() : a controller with no external drivers still integrates and writes its outputs.
    gn = GeneNetworkConnectionist((), _OUT, hidden_size=2)
    assert jnp.asarray(gn.W_in).shape == (3, 0)  # n_gene = 2 latent + 1 observed, n_in = 0
    s = build_state_from_model(gn).init_empty(capacity=2, n_space_dim=2, n_types=1)
    s = s.set('alive', s.alive.at[0].set(True))
    d = gn(s, dt=0.1, key=key)
    assert d.secretion_rate.shape == (2, 1)
    assert np.all(np.isfinite(np.asarray(d.secretion_rate)))


def _assert_grad_finite_and_nonzero(grads):
    leaves = [g for g in jax.tree_util.tree_leaves(grads) if eqx.is_array(g)]
    assert leaves and all(np.all(np.isfinite(np.asarray(g))) for g in leaves)
    assert sum(float(np.abs(np.asarray(g)).sum()) for g in leaves) > 0.0


def test_neural_ode_pathwise_gradient(key):
    mlp = NeuralODE.make_mlp(_IN, _OUT, hidden_size=2, key=key, width=8, depth=1)
    node = NeuralODE(_IN, _OUT, hidden_size=2, mlp=mlp)
    s = build_state_from_model(node).init_empty(capacity=2, n_space_dim=2, n_types=1)
    s = s.set('alive', s.alive.at[0].set(True))
    grads = eqx.filter_grad(lambda m: jnp.sum(m(s, dt=0.1, key=key).secretion_rate))(node)
    _assert_grad_finite_and_nonzero(grads)


def test_gene_network_mwc_pathwise_gradient(key):
    drivers = (StateFieldSpec('maternal', shape=(2,)),)
    genes = (StateFieldSpec('gap', shape=(2,)),)
    gn = GeneNetworkMWC(drivers, genes, hidden_size=1)
    s = build_state_from_model(gn).init_empty(capacity=2, n_space_dim=2, n_types=1)
    s = s.update(
        alive=s.alive.at[0].set(True), maternal=s.maternal.at[0].set(jnp.array([1.0, 0.5]))
    )
    grads = eqx.filter_grad(lambda m: jnp.sum(m(s, dt=0.1, key=key).gap))(gn)
    _assert_grad_finite_and_nonzero(grads)
