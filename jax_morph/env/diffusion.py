import jax
import jax.numpy as np

import jax_md

import equinox as eqx

from .._base import SimulationStep

from typing import Union


class SteadyStateDiffusion(SimulationStep):
    diffusion_coeff: Union[float, jax.Array]
    degradation_rate: Union[float, jax.Array]
    _vmap_diff_inaxes: tuple = eqx.field(static=True)
    _diffusion_type: str = eqx.field(static=True)

    def return_logprob(self) -> bool:
        return False

    def _L_closed_system(self, state):

        # calculate all pairwise distances
        dist = jax_md.space.map_product(jax_md.space.metric(state.displacement))(
            state.position, state.position
        )

        alive = np.where(state.celltype.sum(1) > 0, 1, 0)
        alive = np.outer(alive, alive)

        # zero out connections to inexistent cells
        dist = dist * alive

        # prevent division by zero
        safe_dist = np.where(dist > 0, dist, 1)

        # adjacency matrix
        A = np.where(dist > 0.0, 1 / safe_dist, 0)  # **2

        # graph laplacian
        L = np.diag(np.sum(A, axis=0)) - A

        return L

    def _L_open_system_heur(self, state):

        # calculate all pairwise distances
        dist = jax_md.space.map_product(jax_md.space.metric(state.displacement))(
            state.position, state.position
        )

        alive = np.where(state.celltype.sum(1) > 0, 1, 0)
        alive = np.outer(alive, alive)

        # connect only cells that are nearest neighbors
        nn_dist = state.radius + state.radius.T * alive
        A = np.where(dist < 1.1 * nn_dist, 1, 0) * (1 - np.eye(dist.shape[0])) * alive

        # OPEN BOUNDARY CONDITIONS (1)
        # AWFUL approximation to open boundary conditions
        # Fails miserably if there is no bulk and all cells are connected to the boundary
        # diag = np.sum(A, axis=0).max() * np.eye(A.shape[0]) * alive

        # OPEN BOUNDARY CONDITIONS (2)
        # ALTERNATIVE approx: boundary nodes are the ones that have at least 2 neighbors less than the maximally connected node
        # NOTE: fails to capture early stages of growth, when all cells are on the boundary
        # diag = np.sum(A, axis=0)
        # diag = np.where(diag-diag.max() < -2, diag+1, diag)
        # diag = np.where(np.sum(A, axis=0) > 0, diag, 0)
        # diag = np.diag(diag)

        # OPEN BOUNDARY CONDITIONS (3)
        # YET ANOTHER APPROX: Heuristically bulk nodes have at least 5 neighbors.
        # Hence any node with less than 5 neighbors is a boundary node
        # VERY HEURISTIC but fixes the early growth problem
        diag = np.sum(A, axis=0)
        diag = np.where(diag < 5, diag + 1, diag)
        diag = np.where(np.sum(A, axis=0) > 0, diag, 0)
        diag = np.diag(diag)

        # graph laplacian
        L = diag - A

        return L

    def __init__(
        self,
        *,
        diffusion_coeff=2.0,
        degradation_rate=1.0,
        diffusion_type="approx_open",
        **kwargs
    ):

        self.diffusion_coeff = diffusion_coeff
        self.degradation_rate = degradation_rate

        _inaxes_diffcoef = 0 if np.atleast_1d(self.diffusion_coeff).size > 1 else None
        _inaxes_degrate = 0 if np.atleast_1d(self.degradation_rate).size > 1 else None
        self._vmap_diff_inaxes = (1, _inaxes_diffcoef, _inaxes_degrate, None)

        if diffusion_type in ["closed", "approx_open"]:
            self._diffusion_type = diffusion_type
        else:
            raise ValueError("diffusion_type must be 'closed' or 'approx_open'")

    @jax.named_scope("jax_morph.SteadyStateDiffusion")
    def __call__(self, state, *, key=None, **kwargs):

        # check if degradation rate is cell-specific
        if hasattr(state, "degradation_rate"):
            deg_rate_ = state.degradation_rate
            _, vdiff, _, _ = self._vmap_diff_inaxes
            self._vmap_diff_inaxes = (1, vdiff, 1, None)
        else:
            deg_rate_ = self.degradation_rate

        # calculate graph laplacian
        if self._diffusion_type == "closed":
            L = self._L_closed_system(state)
        elif self._diffusion_type == "approx_open":
            L = self._L_open_system_heur(state)

        # solve for steady state of one chemical
        def _ss_chemfield(P, D, K, L):

            # update laplacian with degradation
            L = D * L + K * np.eye(L.shape[0])

            # solve for steady state
            c = np.linalg.solve(L, P)

            return c

        # calculate steady state chemical field
        _ss_chemfield = jax.vmap(
            _ss_chemfield, in_axes=self._vmap_diff_inaxes, out_axes=1
        )

        new_chem = _ss_chemfield(
            state.secretion_rate, self.diffusion_coeff, deg_rate_, L
        )

        # update chemical field
        state = eqx.tree_at(lambda s: s.chemical, state, new_chem)

        return state


class ExponentialSteadyStateDiffusion(SimulationStep):
    diffusion_coeff: Union[float, jax.Array]
    degradation_rate: Union[float, jax.Array]

    def __init__(
        self,
        diffusion_coeff: Union[float, jax.Array] = 1.0,
        degradation_rate: Union[float, jax.Array] = 1.0,
    ):
        """
        Initialize the ApproxSteadyStateDiffusion simulation step.

        This class calculates the steady-state diffusion of chemicals in an open system with constant uniform degradation.
        Each cell is treated as a point source, and the analytical solution for the steady-state concentration
        of a point source in space is used.

        Args:
            diffusion_coeff (float or jax.Array): Diffusion coefficient(s) for the chemicals.
                Can be a single float for all chemicals or a 1D array with a coefficient for each chemical.
            degradation_rate (float or jax.Array): Degradation rate(s) of chemicals.
                Can be a single float for all chemicals or a 1D array with a rate for each chemical.

        Raises:
            ValueError: If the inputs are not of the correct type (float or jax.Array) or shape (scalar or 1D).
            ValueError: If both inputs are arrays but have different shapes.

        Note:
            If both diffusion_coeff and degradation_rate are provided as arrays, they must have the same shape,
            corresponding to the number of chemicals in the simulation.
        """
        # Check diffusion_coeff
        if isinstance(diffusion_coeff, (float, int)):
            self.diffusion_coeff = float(diffusion_coeff)
        elif isinstance(diffusion_coeff, jax.Array):
            if diffusion_coeff.ndim == 0 or diffusion_coeff.ndim == 1:
                self.diffusion_coeff = diffusion_coeff
            else:
                raise ValueError("diffusion_coeff must be a float or a 1D array")
        else:
            raise ValueError("diffusion_coeff must be a float or a JAX array")

        # Check degradation_rate
        if isinstance(degradation_rate, (float, int)):
            self.degradation_rate = float(degradation_rate)
        elif isinstance(degradation_rate, jax.Array):
            if degradation_rate.ndim == 0 or degradation_rate.ndim == 1:
                self.degradation_rate = degradation_rate
            else:
                raise ValueError("degradation_rate must be a float or a 1D array")
        else:
            raise ValueError("degradation_rate must be a float or a JAX array")

        # Check that if both are arrays, they have the same shape
        if isinstance(self.diffusion_coeff, jax.Array) and isinstance(
            self.degradation_rate, jax.Array
        ):
            if self.diffusion_coeff.shape != self.degradation_rate.shape:
                raise ValueError(
                    "If both diffusion_coeff and degradation_rate are arrays, they must have the same shape"
                )

    def return_logprob(self) -> bool:
        return False

    @staticmethod
    @eqx.filter_jit
    def _compute_steady_state_diffusion(r, sec_rate, deg_rate, diff_coeff):
        """Compute steady-state diffusion for a single chemical."""
        prefactor = sec_rate / (2 * np.sqrt(deg_rate * diff_coeff))
        scaling = np.exp(-r * np.sqrt(deg_rate / diff_coeff))
        return prefactor * scaling

    @eqx.filter_jit
    @jax.named_scope("jax_morph.ApproxSteadyStateDiffusion")
    def __call__(self, state, *, key=None, **kwargs):
        """
        Apply approximate steady-state diffusion to the chemical field.

        Args:
            state: Current simulation state.
            key: Random key (unused in this method).
            **kwargs: Additional keyword arguments.

        Returns:
            Updated simulation state with diffused chemical field.
        """
        # Calculate pairwise distances
        metric = jax_md.space.metric(state.displacement)
        distances = jax_md.space.map_product(metric)(state.position, state.position)

        # Determine vmap axes for diffusion_coeff and degradation_rate
        diff_axis = (
            0
            if isinstance(self.diffusion_coeff, jax.Array)
            and self.diffusion_coeff.ndim > 0
            else None
        )
        deg_axis = (
            0
            if isinstance(self.degradation_rate, jax.Array)
            and self.degradation_rate.ndim > 0
            else None
        )

        # Vectorized function to apply diffusion to all chemicals
        def diffuse_all_chems(sec_rate, diff_coeff, deg_rate):
            concentrations = self._compute_steady_state_diffusion(
                distances,
                sec_rate,
                deg_rate,
                diff_coeff,
            )
            concentrations = concentrations.sum(axis=1)
            return np.where(state.celltype.sum(1) > 0, concentrations, 0.0)

        # Apply diffusion to all chemicals using eqx.filter_vmap
        new_chemical_field = eqx.filter_vmap(
            diffuse_all_chems,
            in_axes=(0, diff_axis, deg_axis),
            out_axes=1,
        )(state.secretion_rate.T, self.diffusion_coeff, self.degradation_rate)

        # Update the chemical field in the state
        return eqx.tree_at(lambda s: s.chemical, state, new_chemical_field)
