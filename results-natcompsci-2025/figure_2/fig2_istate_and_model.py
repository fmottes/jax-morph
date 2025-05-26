# JAX Imports
import jax
import jax.numpy as np

# JAX-Morph Imports
import jax_morph as jxm  # type: ignore
import equinox as eqx


class ZeroDivisionMaskByCellType(jxm.SimulationStep):
    """
    A simulation step that selectively disables cell division for specific cell types.

    This class creates a mask based on cell types and applies it to the division field
    of the cell state, effectively preventing specified cell types from dividing.

    Attributes:
        ctype_zero_div: A list or array indicating which cell types should have division disabled.
                        Each row corresponds to a celltype for which division is disabled.
                        Celltype is a one-hot encoded array of length N_CELL_TYPES.
    """

    ctype_zero_div: jax.Array = eqx.field(static=True)

    def return_logprob(self) -> bool:
        return False

    def __init__(self, state, ctype_zero_div=None):
        if ctype_zero_div is None:
            raise ValueError("ctype_zero_div must be provided")

        if not isinstance(ctype_zero_div, (list, np.ndarray)):
            raise ValueError("ctype_zero_div must be a list or array")

        if len(ctype_zero_div) != state.celltype.shape[1]:
            raise ValueError(
                f"ctype_zero_div must have length {state.celltype.shape[1]} (number of cell types)"
            )

        self.ctype_zero_div = np.asarray(ctype_zero_div).tolist()

    @jax.named_scope("jax_morph.ZeroDivisionMaskByCellType")
    def __call__(self, state, *, key=None, **kwargs):
        # Create mask for cells that should have division zeroed out
        # Multiply celltype matrix by mask vector to get per-cell mask
        div_mask = state.celltype @ np.asarray(self.ctype_zero_div)

        # sum over cell types to get a single mask per cell
        # zero out division for cell types provided in ctype_zero_div
        div_mask = (div_mask[..., None] if div_mask.ndim == 1 else div_mask).sum(
            axis=1
        ) > 0.0

        # Zero out division for masked cells by multiplying by (1 - mask)
        division = state.division * (1.0 - div_mask)[:, None]
        state = eqx.tree_at(lambda s: s.division, state, division)

        return state


# Build initial state
def build_istate(
    N_HIDDEN_GENES=1,
    N_INIT=15,
    N_TOT=150,
    N_DIM=3,
):
    """
    Initialize CellState data structure.

    Parameters:
    - N_HIDDEN_GENES: number of hidden genes
    - N_INIT: number of initial cells
    - N_TOT: total number of cells
    - N_DIM: number of dimensions

    Returns:
    - istate: initial CellState
    """

    # Define number of cell types and chemical species
    # Create CellState data structure

    N_CTYPES = 2
    N_CHEM = 1

    # Define CellState class
    class CellState(jxm.BaseCellState):
        division: jax.Array
        chemical: jax.Array
        secretion_rate: jax.Array
        hidden_state: jax.Array

    disp, shift = jxm.space.free()

    istate = CellState(
        displacement=disp,
        shift=shift,
        position=np.zeros(shape=(N_TOT, N_DIM)),
        celltype=np.zeros(shape=(N_TOT, N_CTYPES)).at[0].set(np.array([1.0, 0.0])),
        radius=np.zeros(shape=(N_TOT, 1)).at[0].set(0.5),
        division=np.zeros(shape=(N_TOT, 1)).at[0].set(1.0),
        chemical=np.zeros(shape=(N_TOT, N_CHEM)),
        secretion_rate=np.zeros(shape=(N_TOT, N_CHEM)).at[0].set(1.0),
        hidden_state=np.zeros(shape=(N_TOT, N_HIDDEN_GENES)),
    )

    # Grow initial cluster of cells (uniform division probability)

    ikey = jax.random.PRNGKey(0)

    mech_potential = jxm.env.mechanics.MorsePotential(epsilon=2.0, alpha=2.8)

    imodel = jxm.Sequential(
        substeps=[
            jxm.env.CellGrowth(growth_rate=0.03, max_radius=0.5, growth_type="linear"),
            jxm.env.mechanics.SGDMechanicalRelaxation(mech_potential),
            jxm.env.CellDivision(),
        ]
    )

    istate = jxm.simulate(imodel, istate, key=ikey, n_steps=N_INIT - 1)[0]

    # Prepare the initial state

    # identify cell idxs with position in the lower half of the y-axis and are alive
    lower_half_idx = np.where(istate.position[:, 1] < 0.5)[0]
    alive_idx = np.where(istate.celltype.sum(1) > 0)[0]
    lower_half_alive_idx = np.intersect1d(lower_half_idx, alive_idx)

    istate = eqx.tree_at(
        lambda s: s.division, istate, istate.division.at[lower_half_alive_idx].set(0.0)
    )
    istate = eqx.tree_at(
        lambda s: s.secretion_rate,
        istate,
        np.zeros_like(istate.secretion_rate).at[lower_half_alive_idx].set(1.0),
    )
    istate = eqx.tree_at(
        lambda s: s.celltype,
        istate,
        istate.celltype.at[lower_half_alive_idx].set(np.array([0.0, 1.0])),
    )

    istate = jxm.env.ExponentialSteadyStateDiffusion(
        diffusion_coeff=0.1,
        degradation_rate=0.1,
    )(istate)

    return istate


# Build the simulation step model
def build_model(init_key, istate):
    """
    Build the simulation step model.

    Parameters:
    - init_key: random key for initialization
    - istate: initial CellState

    """

    # Define mechanical interaction potential
    mech_potential = jxm.env.mechanics.MorsePotential(epsilon=1.7, alpha=3.5)

    # Define model
    model = jxm.Sequential(
        substeps=[
            jxm.env.CellGrowth(growth_rate=0.03, max_radius=0.5, growth_type="linear"),
            jxm.env.mechanics.SGDMechanicalRelaxation(mech_potential),
            jxm.env.ExponentialSteadyStateDiffusion(
                diffusion_coeff=1.0,
                degradation_rate=0.1,
            ),
            jxm.cell.GeneNetwork(
                istate,
                input_fields=["chemical"],
                output_fields=["division"],
                key=init_key,
                transform_output={
                    "division": lambda s, x: x * jax.nn.sigmoid(50 * (s.radius - 0.45))
                },
                degradation_init=jax.nn.initializers.constant(0.1),
                interaction_init=jax.nn.initializers.orthogonal(1e-2),
                dt=0.1,
                T=5.0,
                expression_offset=-0.5,
            ),
            ZeroDivisionMaskByCellType(istate, ctype_zero_div=[0.0, 1.0]),
            jxm.env.CellDivision(),
        ]
    )

    return model
