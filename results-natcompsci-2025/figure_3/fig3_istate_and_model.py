import jax
import jax.numpy as np
import jax_morph as jxm  # type: ignore
import jax_md.space  # type: ignore
import equinox as eqx


from jax.experimental.ode import odeint

from typing import Union, Sequence, Callable


# Gene Network for multiple cell types
class GeneNetwork_ctype(jxm.SimulationStep):
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

    def x_dot(self, xt, t, Input, ct):
        interaction_matrix = np.einsum("ij,jkl->ikl", ct, self.interaction_matrix)
        return (
            jxm.utils.rescaled_algebraic_sigmoid(
                np.einsum("ijk,ik->ij", interaction_matrix, xt),
                shift=0.0,
            )
            - np.atleast_2d(self.degradation_rate) * xt
            + Input
        )

    def circuit_solve(self, x0, Input, ct):

        x_dot_fn = lambda x, t: self.x_dot(x, t, Input, ct)
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

    @jax.named_scope("jax_morph.GeneNetworkCellType")
    @eqx.filter_jit
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
        gene_state = self.circuit_solve(gene_state, Input, state.celltype) * alive

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


def build_istate(init_key, N_CHEM=2, N_HIDDEN=1, ctype_imbalance=0.2):

    # Simulation parameters
    N_DIM = 2
    N_CTYPES = 2
    N_INIT = 20
    N = 120

    # Build initial state
    class CellState(jxm.BaseCellState):
        division: jax.Array
        chemical: jax.Array
        secretion_rate: jax.Array
        hidden_state: jax.Array

    disp, shift = jax_md.space.free()

    istate = CellState(
        displacement=disp,
        shift=shift,
        position=np.zeros(shape=(N, N_DIM)),
        celltype=np.zeros(shape=(N, N_CTYPES)).at[0].set([0.0, 1.0]),
        radius=np.zeros(shape=(N, 1)).at[0].set(0.5),
        division=np.zeros(shape=(N, 1)).at[0].set(1.0),
        chemical=np.zeros(shape=(N, N_CHEM)),
        secretion_rate=np.zeros(shape=(N, N_CHEM)).at[0].set(1.0),
        hidden_state=np.zeros(shape=(N, N_HIDDEN)),
    )

    mech_potential = jxm.env.mechanics.MorsePotential(epsilon=3.0, alpha=2.8)
    init_model = jxm.Sequential(
        substeps=[
            jxm.env.CellDivision(),
            jxm.env.CellGrowth(growth_rate=0.5, max_radius=0.5, growth_type="linear"),
            jxm.env.mechanics.SGDMechanicalRelaxation(mech_potential),
        ]
    )

    init_state, _ = jxm.simulate(init_model, istate, init_key, N_INIT - 1)

    # Assign an initial imbalance in celltypes
    ctype = init_state.celltype.at[: int(ctype_imbalance * N_INIT)].set([1.0, 0.0])
    init_state = eqx.tree_at(lambda s: s.celltype, init_state, ctype)

    return init_state


# Build the simulation step model
def build_model(init_key, istate):

    N_CTYPES = istate.celltype.shape[1]
    N_CHEM = istate.chemical.shape[1]

    ctype_sec_chem = np.zeros((N_CTYPES, N_CHEM))
    ctype_sec_chem = ctype_sec_chem.at[0, : int(N_CHEM / 2)].set(1.0)
    ctype_sec_chem = ctype_sec_chem.at[1, int(N_CHEM / 2) :].set(1.0)

    mech_potential = jxm.env.mechanics.MorsePotential(epsilon=3.0, alpha=2.8)

    model = jxm.Sequential(
        substeps=[
            jxm.env.CellDivision(),
            jxm.env.CellGrowth(growth_rate=0.03, max_radius=0.5, growth_type="linear"),
            jxm.env.mechanics.SGDMechanicalRelaxation(mech_potential),
            jxm.env.SteadyStateDiffusion(
                degradation_rate=1.5,
                diffusion_coeff=0.1,
                diffusion_type="closed",
            ),
            GeneNetwork_ctype(
                istate,
                input_fields=["chemical"],
                output_fields=["secretion_rate", "division"],
                key=init_key,
                transform_output={
                    "division": lambda s, x: x * jax.nn.sigmoid(50 * (s.radius - 0.45))
                },
                expr_level_decay=1.0,
                degradation_rate=jax.nn.initializers.constant(1.0),
                interaction_init=jax.nn.initializers.normal(0.001),
                n_solver_steps=30,
            ),
            jxm.cell.SecretionMaskByCellType(istate, ctype_sec_chem.tolist()),
        ]
    )

    return model
