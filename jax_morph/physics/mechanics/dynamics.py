"""Dynamic (finite-rate) mechanical integrators that move positions under a potential.

``BrownianDynamics`` is one Euler-Maruyama step of overdamped Langevin dynamics: a reparameterized
dynamic stochastic step, both pathwise-differentiable and scorable. ``ActiveBrownianDynamics2D``
adds a persistent, rotationally-diffusing heading and self-propulsion on top of it. Future dynamic
integrators (inertial Langevin, ...) live here alongside them.
"""

import equinox as eqx
import jax
import jax.numpy as jnp

from ...core.ad_utils import gaussian_logp
from ...core.state import POSITION, StateFieldSpec
from ...core.step import StepType, StochasticStep
from .potentials import NoForce, Potential

__all__ = ['BrownianDynamics', 'ActiveBrownianDynamics2D', 'ACTIVE_SPEED', 'ACTIVE_HEADING']


def _gaussian_logp_or_zero(x, mean, std):
    """Gaussian log-density of ``x``, or 0 for a deterministic (``std == 0``) component.

    A zero-noise component (``kT = 0``, or ``rot_diffusion = 0`` for the heading) is deterministic -
    its density is a degenerate delta with no finite log-density and no stochastic information to
    score - so it contributes 0 to the transition log-density. ``kT = 0`` is in fact the standard ABP
    parameterization (pure self-propulsion plus rotational diffusion), so this must not crash. The
    ``std`` is guarded to 1 before ``gaussian_logp`` so its ``1 / std`` / ``log(std)`` never see 0
    (which would give ``NaN`` under ``jax_debug_nans``), and the result is masked back to 0 there.
    """
    positive = std > 0.0
    safe_std = jnp.where(positive, std, 1.0)
    return jnp.where(positive, gaussian_logp(x, mean, safe_std), 0.0)


class BrownianDynamics(StochasticStep):
    r"""Overdamped Langevin (Brownian) dynamics as a reparameterized dynamic stochastic step.

    Each macro-step applies one Euler-Maruyama displacement

    $$\Delta x = -\frac{\nabla U}{\gamma}\,\Delta t + \sqrt{\frac{2\,kT\,\Delta t}{\gamma}}\;\xi,$$

    with $\xi$ standard normal. The noise is recorded, so the step is both **pathwise**-differentiable
    (recompute $\Delta x$ from the frozen noise with live parameters) and **scorable**: ``logp`` is the
    Gaussian log-density of the recorded $\Delta x$ under
    $\mathcal{N}\!\left(-\frac{\nabla U}{\gamma}\Delta t,\ \frac{2\,kT\,\Delta t}{\gamma}\right)$, with
    parameters (the potential's, ``kT``, ``gamma``) recomputed live.

    The trace fields ``{tag}_xi`` and ``{tag}_dx`` are additive dynamic fields (default 0), namespaced
    by ``tag`` so several instances coexist; ``tag='brownian'`` gives ``brownian_xi`` / ``brownian_dx``.

    Attributes:
        potential: Interaction potential supplying mechanical forces.
        n_space_dim: Static spatial dimension; position and trace arrays have trailing shape
            ``(n_space_dim,)``.
        gamma: Translational drag coefficient. Defaults to 1.0.
        kT: Thermal energy controlling translational noise. Defaults to 0.1.
        tag: Static namespace for trace fields. Defaults to ``'brownian'``.
        score_by_default: Whether default trajectory scoring includes this step. Defaults to False.
    """

    step_type = StepType.DYNAMIC
    potential: Potential  # any Potential (its forces give the drift)
    n_space_dim: int = eqx.field(static=True)  # shape of the recorded displacement
    gamma: object  # plain numeric fields: Python float -> static, jax.Array -> traced
    kT: object
    tag: str = eqx.field(static=True)  # namespaces the trace fields

    def __init__(
        self, potential, *, n_space_dim, gamma=1.0, kT=0.1, tag='brownian', score_by_default=False
    ):
        """Build the step from a potential, the spatial dimension, and the Langevin parameters.

        Pass ``potential=None`` for ``NoForce`` (a free Brownian gas: pure noise, no drift).

        Args:
            potential: Interaction potential, or None for `NoForce`.
            n_space_dim: Spatial dimension used for trace-field shapes.
            gamma: Translational drag coefficient. Defaults to 1.0.
            kT: Thermal energy controlling translational noise. Defaults to 0.1.
            tag: Trace-field namespace. Defaults to ``'brownian'``.
            score_by_default: Whether default trajectory scoring includes this step. Defaults to
                False.
        """
        self.potential = potential if potential is not None else NoForce()
        self.n_space_dim = n_space_dim
        self.gamma = gamma
        self.kT = kT
        self.tag = tag
        self.score_by_default = score_by_default

    @property
    def _xi(self):
        return f'{self.tag}_xi'

    @property
    def _dx(self):
        return f'{self.tag}_dx'

    def _check_dim(self, state):
        """Assert the static ``n_space_dim`` matches the state's actual spatial dimension.

        ``n_space_dim`` sizes the recorded ``xi``/``dx`` trace fields at model-build time (before any
        state exists), so a mismatch would otherwise surface as an obscure broadcast error when the
        (N, actual_dim) displacement is written into an (N, n_space_dim) trace field. Both are static
        ints here, so this check runs at trace time and raises before compilation.
        """
        actual = state.position.shape[1]
        if self.n_space_dim != actual:
            raise ValueError(
                f'{type(self).__name__} was built with n_space_dim={self.n_space_dim} but the state '
                f'has spatial dimension {actual}; they must match (n_space_dim sizes the recorded '
                f'{self._xi}/{self._dx} trace fields).'
            )

    def state_reads(self):
        """Reads positions (the drift), plus any state field the potential sources params from."""
        return (POSITION, *self.potential.state_reads())

    def state_writes(self):
        """Writes the position displacement (a dynamic delta)."""
        return (POSITION,)

    def trace_writes(self):
        """The exogenous noise ``{tag}_xi`` and the realized displacement ``{tag}_dx`` (both default 0)."""
        return (
            StateFieldSpec(self._xi, shape=(self.n_space_dim,), heritable=False),
            StateFieldSpec(self._dx, shape=(self.n_space_dim,), heritable=False),
        )

    def _dist(self, state, dt):
        mean = dt * self.potential.forces(state) / self.gamma  # dt * (-grad U / gamma)
        std = jnp.sqrt(2.0 * self.kT * dt / self.gamma)
        return mean, std

    def sample_trace(self, state, *, dt, key):
        """Draw standard-normal displacement noise for pathwise replay.

        Args:
            state: Pre-step state with positions of shape ``(capacity, n_space_dim)``.
            dt: Unused macro-step duration.
            key: JAX PRNG key.

        Returns:
            Trace dictionary containing ``{tag}_xi`` with shape ``(capacity, n_space_dim)``.

        Raises:
            ValueError: If the state's spatial dimension differs from ``n_space_dim``.
        """
        self._check_dim(state)
        return {self._xi: jax.random.normal(key, state.position.shape)}  # noise only

    def replay(self, state, trace, *, dt, pathwise):
        """Apply and record a displacement, recomputing it from noise when pathwise.

        Args:
            state: Live pre-step state.
            trace: Recorded noise and displacement arrays of shape ``(capacity, n_space_dim)``.
            dt: Macro-step duration.
            pathwise: Whether to recompute displacement from the live transition kernel.

        Returns:
            Sparse delta containing ``position`` and the two trace arrays.

        Raises:
            ValueError: If the state's spatial dimension differs from ``n_space_dim``.
        """
        self._check_dim(state)
        xi = trace[self._xi]
        if pathwise:
            mean, std = self._dist(state, dt)
            dx = mean + std * xi  # recompute from the noise with live parameters
        else:
            dx = trace[self._dx]  # frozen realized displacement
        return state.deltas(**{'position': dx, self._xi: xi, self._dx: dx})

    def logp(self, state, trace, dt):
        """Score recorded displacement under the live transition kernel over alive cells.

        Args:
            state: Live pre-step state conditioning the drift and noise scale.
            trace: Recorded displacement trace.
            dt: Macro-step duration.

        Returns:
            Scalar sum of alive-cell Gaussian log-densities, or zero for deterministic noise.
        """
        mean, std = self._dist(state, dt)
        lp = _gaussian_logp_or_zero(trace[self._dx], mean, std)  # elementwise over (cell, dim)
        return jnp.sum(jnp.where(state.alive[:, None], lp, 0.0))


ACTIVE_SPEED = StateFieldSpec('active_speed', heritable=True)  # per-cell self-propulsion speed v0
ACTIVE_HEADING = StateFieldSpec(
    'active_heading', heritable=True
)  # per-cell heading angle theta (shared)


class ActiveBrownianDynamics2D(StochasticStep):
    r"""Overdamped active Brownian (self-propelled) dynamics as a reparameterized dynamic step.

    On top of the passive Langevin drift and noise, each cell self-propels at speed $v_0$ =
    ``active_speed`` along a persistent heading $\theta$, adding $v_0\,(\cos\theta, \sin\theta)$ to the
    drift; the heading rotationally diffuses, $\Delta\theta = \sqrt{2 D_r\,\Delta t}\;\xi_\theta$ at
    rate $D_r$ = ``rot_diffusion``. Both exogenous noises are recorded, so the step is
    pathwise-differentiable and scorable (a Gaussian kernel on the displacement and on the heading
    increment). 2-D only (the heading is an angle).

    Attributes:
        potential: Interaction potential supplying mechanical forces.
        n_space_dim: Static spatial dimension, fixed at 2; position traces have shape
            ``(capacity, 2)``.
        gamma: Translational drag coefficient. Defaults to 1.0.
        kT: Thermal energy controlling translational noise. Defaults to 0.1.
        rot_diffusion: Heading rotational-diffusion coefficient. Defaults to 1.0.
        tag: Static namespace for trace fields. Defaults to ``'active'``.
        score_by_default: Whether default trajectory scoring includes this step. Defaults to False.
    """

    step_type = StepType.DYNAMIC
    potential: Potential
    n_space_dim: int = eqx.field(static=True)  # 2 (v1: the heading is an angle)
    gamma: object
    kT: object
    rot_diffusion: object  # D_r, rotational diffusion of the heading
    tag: str = eqx.field(static=True)

    def __init__(
        self,
        potential,
        *,
        n_space_dim,
        gamma=1.0,
        kT=0.1,
        rot_diffusion=1.0,
        tag='active',
        score_by_default=False,
    ):
        """Build the step from a potential, the (2-D) dimension, and the Langevin / rotational rates.

        Pass ``potential=None`` for ``NoForce`` (self-propulsion + heading noise only, no force);
        this is the natural base for a Vicsek-type model, whose alignment lives in a separate step.

        Args:
            potential: Interaction potential, or None for `NoForce`.
            n_space_dim: Spatial dimension; must be 2.
            gamma: Translational drag coefficient. Defaults to 1.0.
            kT: Thermal energy controlling translational noise. Defaults to 0.1.
            rot_diffusion: Heading rotational-diffusion coefficient. Defaults to 1.0.
            tag: Trace-field namespace. Defaults to ``'active'``.
            score_by_default: Whether default trajectory scoring includes this step. Defaults to
                False.

        Raises:
            ValueError: If ``n_space_dim`` is not 2.
        """
        if n_space_dim != 2:
            raise ValueError(
                'ActiveBrownianDynamics2D is 2-D only: the heading is represented as an angle.'
            )
        self.potential = potential if potential is not None else NoForce()
        self.n_space_dim = n_space_dim
        self.gamma = gamma
        self.kT = kT
        self.rot_diffusion = rot_diffusion
        self.tag = tag
        self.score_by_default = score_by_default

    @property
    def _xi_t(self):
        return f'{self.tag}_xi_t'

    @property
    def _dx(self):
        return f'{self.tag}_dx'

    @property
    def _xi_r(self):
        return f'{self.tag}_xi_r'

    @property
    def _dtheta(self):
        return f'{self.tag}_dtheta'

    def _check_dim(self, state):
        actual = state.position.shape[1]
        if actual != self.n_space_dim:
            raise ValueError(
                f'ActiveBrownianDynamics2D was built with n_space_dim={self.n_space_dim} but the state '
                f'has spatial dimension {actual}; they must match.'
            )

    def state_reads(self):
        """Reads positions (drift), the per-cell speed and heading, and the potential's fields."""
        return (POSITION, ACTIVE_SPEED, ACTIVE_HEADING, *self.potential.state_reads())

    def state_writes(self):
        """Writes the position delta and the shared, persistent (heritable) heading angle."""
        return (POSITION, ACTIVE_HEADING)

    def trace_writes(self):
        """Exogenous noises and realized increments for the translation and the heading (default 0)."""
        return (
            StateFieldSpec(self._xi_t, shape=(self.n_space_dim,), heritable=False),
            StateFieldSpec(self._dx, shape=(self.n_space_dim,), heritable=False),
            StateFieldSpec(self._xi_r, heritable=False),
            StateFieldSpec(self._dtheta, heritable=False),
        )

    def _dist(self, state, dt):
        """Kernel parameters ``(mean_t, std_t, std_r)`` - the single source for sample and score."""
        theta = state.active_heading
        e = jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=-1)  # (N, 2) heading unit vectors
        drift = self.potential.forces(state) / self.gamma + state.active_speed[:, None] * e
        mean_t = dt * drift
        std_t = jnp.sqrt(2.0 * self.kT * dt / self.gamma)
        std_r = jnp.sqrt(2.0 * self.rot_diffusion * dt)
        return mean_t, std_t, std_r

    def sample_trace(self, state, *, dt, key):
        """Draw standard-normal translational and rotational noise for pathwise replay.

        Args:
            state: Pre-step state with positions of shape ``(capacity, 2)``.
            dt: Unused macro-step duration.
            key: JAX PRNG key.

        Returns:
            Trace dictionary with translational noise shape ``(capacity, 2)`` and rotational noise
            shape ``(capacity,)``.

        Raises:
            ValueError: If the state is not two-dimensional.
        """
        self._check_dim(state)
        k_t, k_r = jax.random.split(key)
        return {
            self._xi_t: jax.random.normal(k_t, (state.n_cells, self.n_space_dim)),
            self._xi_r: jax.random.normal(k_r, (state.n_cells,)),
        }

    def replay(self, state, trace, *, dt, pathwise):
        """Apply and record displacement and heading increments from the recorded trace.

        Args:
            state: Live pre-step state.
            trace: Recorded translational and rotational noise/increment arrays.
            dt: Macro-step duration.
            pathwise: Whether to recompute increments from live transition parameters.

        Returns:
            Sparse delta containing position, heading, and trace increments.

        Raises:
            ValueError: If the state is not two-dimensional.
        """
        self._check_dim(state)
        xi_t, xi_r = trace[self._xi_t], trace[self._xi_r]
        if pathwise:
            mean_t, std_t, std_r = self._dist(state, dt)
            dx = mean_t + std_t * xi_t  # live parameters
            dtheta = std_r * xi_r
        else:
            dx, dtheta = trace[self._dx], trace[self._dtheta]  # frozen realized increments
        return state.deltas(
            **{
                'position': dx,
                'active_heading': dtheta,  # shared field; dynamic writers (e.g. an align step) accumulate
                self._xi_t: xi_t,
                self._dx: dx,
                self._xi_r: xi_r,
                self._dtheta: dtheta,
            }
        )

    def logp(self, state, trace, dt):
        """Score recorded displacement and heading increments over alive cells.

        Args:
            state: Live pre-step state conditioning the transition kernels.
            trace: Recorded translational and rotational increments.
            dt: Macro-step duration.

        Returns:
            Scalar sum of alive-cell Gaussian log-densities, with zero contribution for any
            deterministic component.
        """
        mean_t, std_t, std_r = self._dist(state, dt)
        lp_t = _gaussian_logp_or_zero(trace[self._dx], mean_t, std_t)  # (N, 2); 0 if kT == 0
        lp_r = _gaussian_logp_or_zero(
            trace[self._dtheta], 0.0, std_r
        )  # (N,); 0 if rot_diffusion == 0
        alive = state.alive
        return jnp.sum(jnp.where(alive[:, None], lp_t, 0.0)) + jnp.sum(jnp.where(alive, lp_r, 0.0))
