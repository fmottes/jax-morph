import jax
import jax.numpy as np

import jax_morph as jxm  # type: ignore
import jax_md.space  # type: ignore
import equinox as eqx

from typing import Union, Sequence, Callable
from jax.experimental.ode import odeint


# # Rescaled algebraic sigmoid (better gradients - between 0 and 1)
# def rescaled_algebraic_sigmoid(x):
#     return 0.5 + ((x) / (2 * np.sqrt(1 + (x) ** 2)))


class ChemicalField(jxm.SimulationStep):
    field_type: str

    def return_logprob(self) -> bool:
        return False

    def __init__(self, *, field_type="power_law"):
        self.field_type = field_type

    @jax.named_scope("jax_morph.ChemicalField")
    def __call__(self, state, *, key=None, **kwargs):
        if self.field_type == "exp":
            chemfield = self.exp_law(state)
        elif self.field_type == "two_peaks":
            chemfield = self.two_peaks(state)
        elif self.field_type == "off_center":
            chemfield = self.off_center(state)
        else:
            chemfield = self.power_law(state)
        state = eqx.tree_at(lambda s: s.field, state, chemfield)
        return state

    def power_law(self, state):
        center = np.average(state.position, axis=0)
        chemfield = np.linalg.norm(state.position - center, axis=-1)[..., None]
        alive = (state.celltype.sum(-1) > 0)[..., None]
        k = 2.0
        gamma = 0.4  # 80./np.sum(alive)
        # formerly k = 2., gamma = .4
        chemfield = 1.0 / (k + gamma * np.power(chemfield, 2))
        chemfield = np.where(alive, chemfield, 0.0)
        return chemfield

    def exp_law(self, state):
        center = np.average(state.position, axis=0)
        chemfield = np.linalg.norm(state.position - center, axis=-1)[..., None]
        chemfield = np.exp(-chemfield)
        alive = (state.celltype.sum(-1) > 0)[..., None]
        chemfield = np.where(alive, chemfield, 0.0)
        return chemfield

    def two_peaks(self, state):
        center = np.average(state.position, axis=0)
        centers = np.array([[center[0] - 3.0, center[1]], [center[0] + 3.0, center[1]]])

        chemfield1 = np.linalg.norm(state.position - centers[0], axis=-1)[..., None]
        chemfield2 = np.linalg.norm(state.position - centers[1], axis=-1)[..., None]

        chemfield = np.where(chemfield1 < chemfield2, chemfield1, chemfield2)
        chemfield = 100 / (1 + 0.8 * chemfield**2)

        alive = (state.celltype.sum(-1) > 0)[..., None]
        chemfield = np.where(alive, chemfield, 0.0)
        return chemfield

    def off_center(self, state):
        center = np.average(state.position, axis=0)
        center = np.array([center[0] - 3.0, center[1]])

        chemfield = np.linalg.norm(state.position - center, axis=-1)[..., None]
        chemfield = 100 / (2 + 0.8 * np.power(chemfield, 2.0))
        alive = (state.celltype.sum(-1) > 0)[..., None]
        chemfield = np.where(alive, chemfield, 0.0)

        return chemfield


class GeneNetwork(jxm.SimulationStep):
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

    def x_dot(self, xt, t, Input):
        # return jax.nn.sigmoid(xt @ self.interaction_matrix) - np.atleast_2d(self.degradation_rate) * xt + I
        return (
            jxm.utils.rescaled_algebraic_sigmoid(xt @ self.interaction_matrix)
            - np.atleast_2d(self.degradation_rate) * xt
            + Input
        )

    def circuit_solve(self, x0, Input):

        x_dot_fn = lambda x, t: self.x_dot(x, t, Input)
        t = np.linspace(0.0, self.T, self.n_solver_steps)
        x = odeint(x_dot_fn, x0, t)

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
        Input = np.concatenate(
            [
                in_features,
                np.zeros_like(state.hidden_state),
                np.zeros_like(out_features),
            ],
            axis=1,
        )

        alive = np.where(state.celltype.sum(1) > 0.0, 1.0, 0.0)[:, None]
        gene_state = self.circuit_solve(gene_state, Input) * alive

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


def build_istate(init_key, N_CHEM=2, N_HIDDEN=4):

    # Simulation parameters
    N_DIM = 2
    N_CTYPES = 1

    N = 200
    N_INIT = 100

    disp, shift = jax_md.space.free()

    # Build initial state
    class CellState(jxm.BaseCellState):
        division: jax.Array
        chemical: jax.Array
        secretion_rate: jax.Array
        chemical_grad: jax.Array
        hidden_state: jax.Array
        mechanical_stress: jax.Array
        field: jax.Array

    istate = CellState(
        displacement=disp,
        shift=shift,
        position=np.zeros(shape=(N, N_DIM)),
        celltype=np.zeros(shape=(N, N_CTYPES)).at[0].set(1.0),
        radius=np.zeros(shape=(N, 1)).at[0].set(0.5),
        division=np.zeros(shape=(N, 1)).at[0].set(1.0),
        chemical=np.zeros(shape=(N, N_CHEM)),
        chemical_grad=np.zeros(shape=(N, int(N_DIM * N_CHEM))),
        secretion_rate=np.zeros(shape=(N, N_CHEM)).at[0].set(1.0),
        field=np.zeros(shape=(N, 1)),
        mechanical_stress=np.zeros(shape=(N, 1)),
        hidden_state=np.zeros(shape=(N, N_HIDDEN)),
    )

    mech_potential = jxm.env.mechanics.MorsePotential(epsilon=3.0, alpha=2.8)
    init_model = jxm.Sequential(
        substeps=[
            jxm.env.CellDivision(),
            jxm.env.CellGrowth(growth_rate=0.03, max_radius=0.5, growth_type="linear"),
            jxm.env.mechanics.SGDMechanicalRelaxation(
                mech_potential, relaxation_steps=10
            ),
        ]
    )

    init_state, _ = jxm.simulate(init_model, istate, init_key, N_INIT - 1)

    return init_state


# Build the simulation step model
def build_model(init_key, istate, field_type="power_law"):

    mech_potential = jxm.env.mechanics.MorsePotential(epsilon=3.0, alpha=2.8)
    model = jxm.Sequential(
        substeps=[
            jxm.env.CellDivision(),
            jxm.env.CellGrowth(growth_rate=0.03, max_radius=0.5, growth_type="linear"),
            jxm.env.mechanics.SGDMechanicalRelaxation(
                mech_potential, relaxation_steps=10, dt=0.0001
            ),
            jxm.cell.sensing.LocalMechanicalStress(mech_potential),
            ChemicalField(field_type=field_type),
            jxm.env.SteadyStateDiffusion(
                degradation_rate=0.1, diffusion_coeff=0.1, diff_type="approx_open"
            ),
            jxm.cell.sensing.LocalChemicalGradients(),
            GeneNetwork(
                istate,
                input_fields=["chemical", "chemical_grad", "mechanical_stress"],
                output_fields=["secretion_rate", "division"],
                key=init_key,
                transform_output={
                    "division": lambda s, x: x
                    * jax.nn.sigmoid(50 * (s.radius - 0.45))
                    * s.field
                },
                expr_level_decay=0.0,
                interaction_init=jax.nn.initializers.constant(0.0),
                degradation_init=jax.nn.initializers.constant(1.0),
            ),
        ]
    )

    return model
