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


class ApproxSteadyStateDiffusion(SimulationStep):
    diffusion_coeff: Union[float, jax.Array]
    degradation_rate: Union[float, jax.Array]

    def return_logprob(self) -> bool:
        return False

    def __init__(self, *, diffusion_coeff=2.0, degradation_rate=1.0, **kwargs):
        """
        OLD APPROACH:
        Diffusion approximated by sum of exponential chemical profiles.
        """

        self.diffusion_coeff = diffusion_coeff
        self.degradation_rate = degradation_rate

    @jax.named_scope("jax_morph.ApproxSteadyStateDiffusion")
    def __call__(self, state, *, key=None, **kwargs):

        def diffuse_onechem_ss(r, secRate, degRate, diffCoeff):
            diff = (
                secRate
                / (2 * np.sqrt(degRate * diffCoeff))
                * np.exp(-r * np.sqrt(degRate / diffCoeff))
            )

            return diff

        metric = jax_md.space.metric(state.displacement)
        d = jax_md.space.map_product(metric)

        # calculate all pairwise distances
        dist = d(state.position, state.position)

        # loop over all chemicals (vmap in future)
        new_chem = []
        for i in np.arange(state.secretion_rate.shape[1]):

            c = diffuse_onechem_ss(
                dist,
                state.secretion_rate[:, i],
                self.degradation_rate,
                self.diffusion_coeff,
            )

            c = c.sum(axis=1)

            # zero out concentration on empty sites
            c = np.where(state.celltype.sum(1) > 0, c, 0.0)

            c = np.reshape(c, (-1, 1))  # make into column vector
            new_chem.append(c)

        new_chem = np.hstack(new_chem)

        # update chemical field
        state = eqx.tree_at(lambda s: s.chemical, state, new_chem)

        return state
