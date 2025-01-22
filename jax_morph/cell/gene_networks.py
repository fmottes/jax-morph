import jax
import jax.numpy as np

from jax.experimental.ode import odeint
import diffrax
import equinox as eqx

from .._base import SimulationStep
from ..utils import rescaled_algebraic_sigmoid

from typing import Union, Sequence, Callable


class GeneNetwork(SimulationStep):
    """A gene regulatory network simulation step.

    This class implements a gene regulatory network model that evolves gene expression levels
    over time according to a system of ordinary differential equations (ODEs).

    The interaction matrix is defined as W (i -> j), where i is the source gene and j is the target gene.

    Attributes:
        input_fields: Names of input fields from the state to use as inputs
        output_fields: Names of output fields to write results to
        out_indices: Indices mapping network outputs to output fields
        transform_output: Optional transforms to apply to outputs before writing to state
        dt: Time step size for ODE solver
        T: Total simulation time
        interaction_matrix: Matrix of gene-gene interaction weights
        degradation_rate: Per-gene degradation rates
        expression_offset: Offset added to interaction terms

    The network dynamics are governed by the equation:
        dx/dt = σ(Wx + b) - γx + I

    where:
        x: Gene expression levels
        W: interaction_matrix (i -> j)
        b: expression_offset
        γ: degradation_rate
        I: External inputs
        σ: Rescaled algebraic sigmoid activation
    """

    input_fields: Sequence[str] = eqx.field(static=True)
    output_fields: Sequence[str] = eqx.field(static=True)
    out_indices: tuple = eqx.field(static=True)
    transform_output: Union[Callable, None] = eqx.field(static=True)
    dt: float = eqx.field(static=True)
    T: float = eqx.field(static=True)
    interaction_matrix: jax.Array
    degradation_rate: Union[float, jax.Array]
    expression_offset: Union[float, jax.Array]

    def return_logprob(self) -> bool:
        return False

    def vector_field(self, t, x, args):

        Inputs = args
        interactions = rescaled_algebraic_sigmoid(
            self.interaction_matrix.T @ x + self.expression_offset
        )
        degradation = np.atleast_2d(self.degradation_rate) * x

        return interactions - degradation + Inputs

    def circuit_solve(self, x0, Inputs):
        term = diffrax.ODETerm(self.vector_field)
        solver = diffrax.Dopri5()
        t0 = 0.0
        t1 = self.T
        dt0 = self.dt
        saveat = diffrax.SaveAt(ts=[t1])

        solution = diffrax.diffeqsolve(
            term,
            solver,
            t0,
            t1,
            dt0,
            x0,
            args=Inputs,
            saveat=saveat,
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
        )

        return solution.ys[0]

    def __init__(
        self,
        state,
        input_fields,
        output_fields,
        *,
        key,
        interaction_init=jax.nn.initializers.normal(0.001),
        degradation_init=jax.nn.initializers.constant(0.1),
        expression_offset=0.0,
        transform_output=None,
        T=1.0,
        dt=0.1,
        **kwargs
    ):
        self.input_fields = input_fields
        self.output_fields = output_fields

        self.dt = dt
        self.T = T

        if self.input_fields:
            in_shape = np.concatenate(
                [getattr(state, field) for field in input_fields], axis=1
            ).shape[-1]
        else:
            in_shape = 0

        out_shape = np.concatenate(
            [getattr(state, field) for field in output_fields], axis=1
        ).shape[-1]

        system_size = int(in_shape + state.hidden_state.shape[-1] + out_shape)

        self.interaction_matrix = interaction_init(
            key, shape=(system_size, system_size)
        )

        self.degradation_rate = degradation_init(key, shape=(1, system_size)).tolist()

        self.expression_offset = expression_offset * np.ones((1, system_size))

        out_sizes = [getattr(state, field).shape[-1] for field in self.output_fields]

        self.out_indices = tuple(
            (system_size - np.cumsum(np.asarray(out_sizes)[::-1])).tolist()[::-1]
            + [system_size]
        )

        self.transform_output = dict(
            zip(self.output_fields, [None] * len(self.output_fields))
        )

        if transform_output is not None:
            self.transform_output.update(transform_output)

    @jax.named_scope("jax_morph.GeneNetwork")
    def __call__(self, state, *, key=None, **kwargs):

        if self.input_fields:
            in_features = np.concatenate(
                [getattr(state, field) for field in self.input_fields], axis=1
            )
        else:
            in_features = np.empty((state.hidden_state.shape[0], 0))

        out_features = np.concatenate(
            [getattr(state, field) for field in self.output_fields], axis=1
        )

        gene_state = np.concatenate(
            [
                in_features,
                state.hidden_state,
                out_features,
            ],
            axis=1,
        )

        Inputs = np.concatenate(
            [
                in_features,
                np.zeros_like(state.hidden_state),
                np.zeros_like(out_features),
            ],
            axis=1,
        )

        alive = np.where(state.celltype.sum(1) > 0.0, 1.0, 0.0)[:, None]
        gene_state = self.circuit_solve(gene_state, Inputs) * alive

        hidden_state = gene_state[
            :,
            in_features.shape[-1] : in_features.shape[-1]
            + state.hidden_state.shape[-1],
        ]

        state = eqx.tree_at(lambda s: s.hidden_state, state, hidden_state)

        for i, field in enumerate(self.output_fields):
            new_field = gene_state[:, self.out_indices[i] : self.out_indices[i + 1]]

            if self.transform_output[field] is not None:
                new_field = self.transform_output[field](state, new_field)

            new_field = new_field * alive

            state = eqx.tree_at(lambda s: getattr(s, field), state, new_field)

        return state


################################################################################
# OLD Gene Network
################################################################################
class OLD_GeneNetwork(SimulationStep):
    input_fields: Sequence[str] = eqx.field(static=True)
    output_fields: Sequence[str] = eqx.field(static=True)
    out_indices: tuple = eqx.field(static=True)
    transform_output: Union[Callable, None] = eqx.field(static=True)
    n_solver_steps: int = eqx.field(static=True)
    dt: float = eqx.field(static=True)
    T: float = eqx.field(static=True)
    interaction_matrix: jax.Array
    degradation_rate: Union[float, jax.Array]
    expr_level_decay: Union[float, jax.Array]

    def return_logprob(self) -> bool:
        return False

    def x_dot(self, xt, t, I):
        # return jax.nn.sigmoid(xt @ self.interaction_matrix) - np.atleast_2d(self.degradation_rate) * xt + I
        return (
            rescaled_algebraic_sigmoid(xt @ self.interaction_matrix)
            - np.atleast_2d(self.degradation_rate) * xt
            + I
        )

    def circuit_solve(self, x0, I):

        t = np.linspace(0.0, self.T, self.n_solver_steps)

        x = odeint(self.x_dot, x0, t, I)

        return x[-1]

    def __init__(
        self,
        state,
        input_fields,
        output_fields,
        *,
        key,
        expr_level_decay=0.0,
        interaction_init=jax.nn.initializers.normal(1.0),
        degradation_init=jax.nn.initializers.constant(0.1),
        transform_output=None,
        n_solver_steps=int(1e2),
        dt=0.1,
        **kwargs
    ):

        self.input_fields = input_fields
        self.output_fields = output_fields
        self.expr_level_decay = float(expr_level_decay)

        self.n_solver_steps = int(n_solver_steps)
        self.dt = dt
        self.T = float(n_solver_steps * dt)

        in_shape = np.concatenate(
            [getattr(state, field) for field in input_fields], axis=1
        ).shape[-1]
        out_shape = np.concatenate(
            [getattr(state, field) for field in output_fields], axis=1
        ).shape[-1]

        system_size = int(in_shape + state.hidden_state.shape[-1] + out_shape)

        self.interaction_matrix = interaction_init(
            key, shape=(system_size, system_size)
        )
        self.degradation_rate = degradation_init(key, shape=(1, system_size)).tolist()

        out_sizes = [getattr(state, field).shape[-1] for field in self.output_fields]
        self.out_indices = tuple(
            (system_size - np.cumsum(np.asarray(out_sizes)[::-1])).tolist()[::-1]
            + [system_size]
        )

        self.transform_output = dict(
            zip(self.output_fields, [None] * len(self.output_fields))
        )

        if transform_output is not None:
            self.transform_output.update(transform_output)

    @jax.named_scope("jax_morph.GeneNetwork")
    def __call__(self, state, *, key=None, **kwargs):

        # concatenate input features
        in_features = np.concatenate(
            [getattr(state, field) for field in self.input_fields], axis=1
        )
        out_features = np.concatenate(
            [getattr(state, field) for field in self.output_fields], axis=1
        )

        gene_state = np.concatenate(
            [
                in_features,
                (1 - self.expr_level_decay) * state.hidden_state,
                out_features,
            ],
            axis=1,
        )
        I = np.concatenate(
            [
                in_features,
                np.zeros_like(state.hidden_state),
                np.zeros_like(out_features),
            ],
            axis=1,
        )

        alive = np.where(state.celltype.sum(1) > 0.0, 1.0, 0.0)[:, None]
        gene_state = self.circuit_solve(gene_state, I) * alive

        hidden_state = gene_state[
            :,
            in_features.shape[-1] : in_features.shape[-1]
            + state.hidden_state.shape[-1],
        ]

        # update state
        state = eqx.tree_at(lambda s: s.hidden_state, state, hidden_state)

        # update output
        for i, field in enumerate(self.output_fields):

            new_field = gene_state[:, self.out_indices[i] : self.out_indices[i + 1]]

            if self.transform_output[field] is not None:
                new_field = self.transform_output[field](state, new_field)

            new_field = new_field * alive

            state = eqx.tree_at(lambda s: getattr(s, field), state, new_field)

        return state


# Gene Network for multiple cell types
class OLD_GeneNetwork_ctype(SimulationStep):
    input_fields: Sequence[str] = eqx.field(static=True)
    output_fields: Sequence[str] = eqx.field(static=True)
    out_indices: tuple = eqx.field(static=True)
    transform_output: Union[Callable, None] = eqx.field(static=True)
    n_solver_steps: int = eqx.field(static=True)
    dt: float = eqx.field(static=True)
    T: float = eqx.field(static=True)
    interaction_matrix: jax.Array
    degradation_rate: Union[float, jax.Array]
    expr_level_decay: Union[float, jax.Array]

    def return_logprob(self) -> bool:
        return False

    def x_dot(self, xt, t, I, ct):
        interaction_matrix = np.einsum("ij,jkl->ikl", ct, self.interaction_matrix)
        # return jax.nn.sigmoid(xt @ self.interaction_matrix) - np.atleast_2d(self.degradation_rate) * xt + I
        return (
            rescaled_algebraic_sigmoid(np.einsum("ijk,ik->ij", interaction_matrix, xt))
            - np.atleast_2d(self.degradation_rate) * xt
            + I
        )

    def circuit_solve(self, x0, I, ct):
        t = np.linspace(0.0, self.T, self.n_solver_steps)

        x = odeint(self.x_dot, x0, t, I, ct)

        return x[-1]

    def __init__(
        self,
        state,
        input_fields,
        output_fields,
        *,
        key,
        expr_level_decay=0.0,
        interaction_init=jax.nn.initializers.normal(1.0),
        degradation_init=jax.nn.initializers.constant(0.1),
        transform_output=None,
        n_solver_steps=int(1e2),
        dt=0.1,
        **kwargs
    ):

        self.input_fields = input_fields
        self.output_fields = output_fields
        self.expr_level_decay = float(expr_level_decay)

        self.n_solver_steps = int(n_solver_steps)
        self.dt = dt
        self.T = float(n_solver_steps * dt)

        in_shape = np.concatenate(
            [getattr(state, field) for field in input_fields], axis=1
        ).shape[-1]
        out_shape = np.concatenate(
            [getattr(state, field) for field in output_fields], axis=1
        ).shape[-1]

        system_size = int(in_shape + state.hidden_state.shape[-1] + out_shape)

        n_ctypes = state.celltype.shape[1]
        self.interaction_matrix = interaction_init(
            key, shape=(n_ctypes, system_size, system_size)
        )
        self.degradation_rate = degradation_init(key, shape=(1, system_size)).tolist()

        out_sizes = [getattr(state, field).shape[-1] for field in self.output_fields]
        self.out_indices = tuple(
            (system_size - np.cumsum(np.asarray(out_sizes)[::-1])).tolist()[::-1]
            + [system_size]
        )

        self.transform_output = dict(
            zip(self.output_fields, [None] * len(self.output_fields))
        )

        if transform_output is not None:
            self.transform_output.update(transform_output)

    @jax.named_scope("jax_morph.GeneNetwork")
    def __call__(self, state, *, key=None, **kwargs):

        # concatenate input features
        in_features = np.concatenate(
            [getattr(state, field) for field in self.input_fields], axis=1
        )
        out_features = np.concatenate(
            [getattr(state, field) for field in self.output_fields], axis=1
        )

        gene_state = np.concatenate(
            [
                in_features,
                (1 - self.expr_level_decay) * state.hidden_state,
                out_features,
            ],
            axis=1,
        )
        cell_type = state.celltype
        I = np.concatenate(
            [
                in_features,
                np.zeros_like(state.hidden_state),
                np.zeros_like(out_features),
            ],
            axis=1,
        )

        alive = np.where(state.celltype.sum(1) > 0.0, 1.0, 0.0)[:, None]
        gene_state = self.circuit_solve(gene_state, I, cell_type) * alive

        hidden_state = gene_state[
            :,
            in_features.shape[-1] : in_features.shape[-1]
            + state.hidden_state.shape[-1],
        ]

        # update state
        state = eqx.tree_at(lambda s: s.hidden_state, state, hidden_state)

        # update output
        for i, field in enumerate(self.output_fields):

            new_field = gene_state[:, self.out_indices[i] : self.out_indices[i + 1]]

            if self.transform_output[field] is not None:
                new_field = self.transform_output[field](state, new_field)

            new_field = new_field * alive

            state = eqx.tree_at(lambda s: getattr(s, field), state, new_field)

        return state
