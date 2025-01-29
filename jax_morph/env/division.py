import jax
import jax.numpy as np

import equinox as eqx

from .._base import SimulationStep


class CellDivision(SimulationStep):
    birth_radius_multiplier: float
    straight_through: bool = eqx.static_field()

    def return_logprob(self) -> bool:
        return not self.straight_through

    def __init__(
        self, *, birth_radius_multiplier="auto", straight_through=False, **kwargs
    ):

        self.birth_radius_multiplier = birth_radius_multiplier

        self.straight_through = straight_through

    @jax.named_scope("jax_morph.CellDivision")
    def __call__(self, state, *, key=None, **kwargs):

        idx_new_cell = np.count_nonzero(state.celltype.sum(1))

        # Determine dimensionality dynamically
        dimension = state.position.shape[1]

        # # Adjust birth radius multiplier to have half the volume after division
        if self.birth_radius_multiplier == "auto":
            birth_radius_multiplier = float(2 ** (-1 / dimension))

        # Split key
        subkey_div, subkey_angle = jax.random.split(key)

        p = state.division.squeeze()
        p = p / p.sum()

        idx_dividing_cell = jax.random.choice(subkey_div, np.arange(p.shape[0]), p=p)

        onehot = jax.nn.one_hot(idx_dividing_cell, state.celltype.shape[0])

        if self.straight_through:
            new_cell_contribs = onehot + p - jax.lax.stop_gradient(p)
        else:
            new_cell_contribs = onehot

        division_matrix = (
            np.eye(state.celltype.shape[0]).at[idx_new_cell].set(new_cell_contribs)
        )

        state = jax.tree_map(lambda x: np.dot(division_matrix, x), state)

        # Resize cell radii
        resize_rad = state.radius.at[idx_new_cell].set(
            state.radius[idx_new_cell] * birth_radius_multiplier
        )
        resize_rad = resize_rad.at[idx_dividing_cell].set(
            state.radius[idx_dividing_cell] * birth_radius_multiplier
        )

        state = eqx.tree_at(lambda s: s.radius, state, resize_rad)

        # Position of new cells
        if dimension == 2:

            angle = jax.random.uniform(subkey_angle, minval=0.0, maxval=2 * np.pi)
            cell_displacement = (
                state.radius[idx_dividing_cell]
                * birth_radius_multiplier
                * np.array([np.cos(angle), np.sin(angle)])
            )

        elif dimension == 3:

            angle_keys = jax.random.split(subkey_angle, 2)
            phi = jax.random.uniform(angle_keys[0], minval=0.0, maxval=2 * np.pi)
            theta = jax.random.uniform(angle_keys[1], minval=0.0, maxval=np.pi)
            r = state.radius[idx_dividing_cell] * birth_radius_multiplier

            cell_displacement = r * np.array(
                [
                    np.sin(theta) * np.cos(phi),
                    np.sin(theta) * np.sin(phi),
                    np.cos(theta),
                ]
            )

        else:
            raise ValueError("Unsupported dimension: must be 2 or 3")

        new_position = state.position.at[idx_new_cell].set(
            state.position[idx_dividing_cell] - cell_displacement
        )
        new_position = new_position.at[idx_dividing_cell].set(
            state.position[idx_dividing_cell] + cell_displacement
        )

        state = eqx.tree_at(lambda s: s.position, state, new_position)

        if self.straight_through:
            return state
        else:
            return state, np.log(p[idx_dividing_cell])


class CellDivisionReparam(SimulationStep):
    birth_radius_multiplier: float
    softmax_T: float

    def return_logprob(self) -> bool:
        return False

    def __init__(
        self, *, birth_radius_multiplier=float(1 / np.sqrt(2)), softmax_T=1.0, **kwargs
    ):

        self.birth_radius_multiplier = birth_radius_multiplier
        self.softmax_T = softmax_T if softmax_T > 0.0 else 1e-3

    def _gumbel_logits_to_onehot(self, logits):

        onehot = jax.nn.one_hot(np.argmax(logits), logits.shape[-1])

        logits = jax.nn.softmax(logits)

        # straight through
        zero = logits - jax.lax.stop_gradient(logits)

        return zero + jax.lax.stop_gradient(onehot)

    @jax.named_scope("jax_morph.CellDivisionReparam")
    def __call__(self, state, *, key=None, **kwargs):

        idx_new_cell = np.count_nonzero(state.celltype.sum(1))

        # split key
        subkey_div, subkey_place = jax.random.split(key, 2)

        p = state.division
        p = p.squeeze()

        safe_p = np.where(p > 0.0, p, 1)
        logp = np.where(p > 0.0, np.log(safe_p), 0.0)

        logits = (
            logp + jax.random.gumbel(subkey_div, shape=logp.shape)
        ) / self.softmax_T
        logits = np.where(p > 0.0, logits, -np.inf)

        # GUMBEL SOFTMAX STRAIGHT THROUGH step
        new_cell_contribs = self._gumbel_logits_to_onehot(logits)

        idx_dividing_cell = np.argmax(new_cell_contribs)

        division_matrix = (
            np.eye(state.celltype.shape[0]).at[idx_new_cell].set(new_cell_contribs)
        )

        state = jax.tree_map(lambda x: np.dot(division_matrix, x), state)

        # resize cell radii
        resize_rad = state.radius.at[idx_new_cell].set(
            state.radius[idx_new_cell] * self.birth_radius_multiplier
        )
        resize_rad = resize_rad.at[idx_dividing_cell].set(
            state.radius[idx_dividing_cell] * self.birth_radius_multiplier
        )

        state = eqx.tree_at(lambda s: s.radius, state, resize_rad)

        # POSITION OF NEW CELLS
        angle = jax.random.uniform(subkey_place, minval=0.0, maxval=2 * np.pi)

        cell_displacement = (
            state.radius[idx_dividing_cell]
            * self.birth_radius_multiplier
            * np.array([np.cos(angle), np.sin(angle)])
        )

        new_position = state.position.at[idx_new_cell].set(
            state.position[idx_dividing_cell] - cell_displacement
        )
        new_position = new_position.at[idx_dividing_cell].set(
            new_position[idx_dividing_cell] + cell_displacement
        )

        state = eqx.tree_at(lambda s: s.position, state, new_position)

        return state


class IndependentCellDivision(SimulationStep):
    birth_radius_multiplier: float
    straight_through: eqx.field(static=True)

    def return_logprob(self) -> bool:
        return not self.straight_through

    def __init__(
        self,
        *,
        birth_radius_multiplier=float(1 / np.sqrt(2)),
        straight_through=False,
        **kwargs
    ):

        self.birth_radius_multiplier = birth_radius_multiplier
        self.straight_through = straight_through

    @jax.named_scope("jax_morph.CellDivision")
    def __call__(self, state, *, key=None, **kwargs):

        idx_new_cell = np.count_nonzero(state.celltype.sum(1))

        # split key
        subkey_div, subkey_place = jax.random.split(key, 2)

        p = state.division.squeeze()

        div_sample = jax.random.bernoulli(subkey_div, p)

        idx_dividing_cells = np.nonzero(div_sample, size=p.shape[0], fill_value=-1)[0]

        onehot = jax.nn.one_hot(idx_dividing_cells, state.celltype.shape[0])

        if self.straight_through:
            new_cell_contribs = onehot + p - jax.lax.stop_gradient(p)
        else:
            new_cell_contribs = onehot

        idx_new_cells = idx_new_cell + np.arange(len(idx_dividing_cells))

        division_matrix = (
            np.eye(state.celltype.shape[0]).at[idx_new_cells].set(new_cell_contribs)
        )

        state = jax.tree_map(lambda x: np.dot(division_matrix, x), state)

        # resize cell radii
        resize_rad = state.radius.at[idx_new_cells].set(
            state.radius[idx_new_cells] * self.birth_radius_multiplier
        )
        resize_rad = resize_rad.at[idx_dividing_cells].set(
            state.radius[idx_dividing_cells] * self.birth_radius_multiplier
        )

        state = eqx.tree_at(lambda s: s.radius, state, resize_rad)

        # POSITION OF NEW CELLS
        angle = jax.random.uniform(
            subkey_place,
            shape=(len(idx_dividing_cells), 1),
            minval=0.0,
            maxval=2 * np.pi,
        )

        cell_displacement = (
            state.radius[idx_dividing_cells]
            * self.birth_radius_multiplier
            * np.hstack([np.cos(angle), np.sin(angle)])
        )

        new_position = state.position.at[idx_new_cells].set(
            state.position[idx_dividing_cells] - cell_displacement
        )
        new_position = new_position.at[idx_dividing_cells].set(
            new_position[idx_dividing_cells] + cell_displacement
        )

        state = eqx.tree_at(lambda s: s.position, state, new_position)

        if self.straight_through:
            return state

        else:
            safe_p = np.where(p > 0.0, p, 0.5)

            safe_logp = np.where(p > 0.0, np.log(safe_p), jax.lax.stop_gradient(0.0))
            alive_1mlogp = np.where(
                p > 0, np.log(1 - safe_p), jax.lax.stop_gradient(0.0)
            )

            logp = np.where(div_sample > 0, safe_logp, alive_1mlogp)

            return state, np.sum(logp)
