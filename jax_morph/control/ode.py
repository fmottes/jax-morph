"""Continuous-time cell controllers implemented as per-cell ODEs."""

import abc
import math

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp

from ..core.state import StateFieldSpec
from ..core.step import SimulationStep, StepType

__all__ = [
    'ODEController',
    'GeneNetworkConnectionist',
    'NeuralODE',
    'GeneNetworkMWC',
]


def _width(spec):
    """Return the flattened per-cell width of a state field spec."""
    return math.prod(spec.shape) if spec.shape else 1


def _rescaled_sigmoid(x):
    """Return the algebraic sigmoid without overflowing ``x * x`` at large drives."""
    limit = jnp.finfo(x.dtype).max
    finite_x = jnp.clip(x, -limit, limit)
    scale = jnp.maximum(1.0, jnp.abs(finite_x))
    scaled_x = finite_x / scale
    scaled_one = 1.0 / scale
    ratio = scaled_x / jnp.sqrt(scaled_one * scaled_one + scaled_x * scaled_x)
    return 0.5 + 0.5 * ratio


def _positive_from_log(log_value):
    """Exponentiate a log-parameter within its dtype's finite positive range."""
    info = jnp.finfo(log_value.dtype)
    min_log = math.log(float(info.tiny))
    max_log = math.log(float(info.max)) - math.log(4.0)
    return jnp.exp(jnp.clip(log_value, min_log, max_log))


class ODEController(SimulationStep):
    """Abstract dynamic controller that integrates a per-cell ODE over one macro-step.

    Subclasses declare their parameter fields and implement ``vector_field``. A controller's evolving
    state is the concatenation of a latent hidden state and its output fields; ``hidden_size=0`` yields
    a pure-output ODE without allocating a hidden-state field. Constructors take parameter values
    directly and do no random initialization - unset parameters default to zeros from the field specs.

    Attributes:
        input_specs: Static tuple of fixed driver-field specifications.
        output_specs: Static tuple of observed evolving output-field specifications.
        hidden_size: Static latent-state width; hidden arrays have shape ``(capacity, hidden_size)``.
        in_size: Static flattened width of all input fields.
        out_size: Static flattened width of all output fields.
        tag: Static namespace for the optional latent state field.

    Methods:
        vector_field: Evaluate the per-cell ODE derivative.
        __call__: Integrate the ODE for one macro-step.
    """

    step_type = StepType.DYNAMIC
    input_specs: tuple = eqx.field(static=True)
    output_specs: tuple = eqx.field(static=True)
    hidden_size: int = eqx.field(static=True)
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)
    tag: str = eqx.field(static=True)

    def __init__(self, input_specs, output_specs, hidden_size, *, tag):
        """Record field specifications and derive flattened input and output widths.

        Args:
            input_specs: Per-cell fields held fixed while integrating the ODE.
            output_specs: Per-cell fields that form the observed evolving state.
            hidden_size: Width of the controller's latent evolving state.
            tag: Namespace for the latent state field.

        Raises:
            ValueError: If ``hidden_size`` is negative.
        """
        if hidden_size < 0:
            raise ValueError(f'hidden_size must be non-negative, got {hidden_size}')
        self.input_specs = tuple(input_specs)
        self.output_specs = tuple(output_specs)
        self.hidden_size = hidden_size
        self.in_size = sum(_width(spec) for spec in self.input_specs)
        self.out_size = sum(_width(spec) for spec in self.output_specs)
        self.tag = tag

    def _resolve_param(self, value, shape, name):
        """Return an explicit parameter, filling ``None`` with zeros and shape-checking the rest.

        A ``None`` value becomes ``jnp.zeros(shape)`` (a traced, optimizable array). Any other value
        is kept verbatim so a Python list or tuple stays a static, frozen leaf while a ``jax.Array``
        stays traced; its shape is validated against the field spec.

        Args:
            value: The supplied parameter, or None to use the zero default.
            shape: The required parameter shape.
            name: The parameter name, used in the shape-mismatch message.

        Returns:
            The zero default or the original value.

        Raises:
            ValueError: If an explicit parameter does not have the required shape.
        """
        if value is None:
            return jnp.zeros(shape)
        resolved_shape = jnp.asarray(value).shape
        if resolved_shape != shape:
            raise ValueError(
                f'{type(self).__name__}.{name} must have shape {shape}, got {resolved_shape}'
            )
        return value

    @abc.abstractmethod
    def vector_field(self, t, evolving, inputs):
        """Return the derivative of the evolving block with fixed per-cell inputs.

        Args:
            t: Physical integration time within the macro-step.
            evolving: Batched concatenation of hidden and output state.
            inputs: Batched, fixed driver fields.

        Returns:
            Batched derivative with the same shape as ``evolving``.
        """
        raise NotImplementedError

    @property
    def _hidden(self):
        """Return the namespaced latent-state field name."""
        return f'{self.tag}_hidden'

    def state_reads(self):
        """Return the driver fields read by the controller."""
        return self.input_specs

    def state_writes(self):
        """Return output fields and, when present, the heritable latent state field."""
        hidden = (
            (StateFieldSpec(self._hidden, shape=(self.hidden_size,)),) if self.hidden_size else ()
        )
        return (*hidden, *self.output_specs)

    def _pack(self, state, specs):
        """Flatten and concatenate named per-cell state fields in declaration order."""
        if not specs:
            return jnp.zeros((state.n_cells, 0), dtype=state.position.dtype)
        return jnp.concatenate(
            [state[spec.name].reshape(state.n_cells, -1) for spec in specs], axis=1
        )

    def __call__(self, state, *, dt, key):
        """Integrate over ``dt`` and return a sparse state delta.

        The random key is accepted for the common simulation-step interface but is unused.

        Args:
            state: Input state containing all input, output, and optional hidden fields.
            dt: Macro-step duration to integrate.
            key: Unused JAX PRNG key.

        Returns:
            Sparse delta state with the controller's output and optional hidden-field increments.
        """
        del key
        n = state.n_cells
        inputs = self._pack(state, self.input_specs)
        outputs = self._pack(state, self.output_specs)
        hidden = state[self._hidden] if self.hidden_size else jnp.zeros((n, 0), dtype=outputs.dtype)
        y0 = jnp.concatenate([hidden, outputs], axis=1)
        term = diffrax.ODETerm(lambda t, y, args: self.vector_field(t, y, inputs))
        solution = diffrax.diffeqsolve(
            term,
            diffrax.Dopri5(),
            t0=0.0,
            t1=dt,
            dt0=dt,
            y0=y0,
            stepsize_controller=diffrax.PIDController(rtol=1e-4, atol=1e-6),
            saveat=diffrax.SaveAt(t1=True),
        )
        delta = solution.ys[-1] - y0
        deltas = {self._hidden: delta[:, : self.hidden_size]} if self.hidden_size else {}
        index = self.hidden_size
        for spec in self.output_specs:
            width = _width(spec)
            chunk = delta[:, index : index + width]
            index += width
            deltas[spec.name] = chunk[:, 0] if spec.shape == () else chunk.reshape((n, *spec.shape))
        return state.deltas(**deltas)


class GeneNetworkConnectionist(ODEController):
    r"""Connectionist gene circuit: sigmoid-saturated production from a linear regulatory drive.

    Each gene concentration $g$ evolves as production minus linear degradation,

    $$\dot{g} = \sigma\!\left(W_\mathrm{gene}\, g + W_\mathrm{in}\, u + b\right) - \gamma\, g,$$

    where $u$ are the driver inputs, $\sigma$ is a saturating sigmoid, and $\gamma$ is the per-gene
    degradation rate. Latent (hidden) genes, when present, evolve alongside the observed outputs and
    regulate every gene in the circuit.

    Attributes:
        W_gene: Gene interaction matrix of shape ``(n_gene, n_gene)``.
        W_in: Input mixing matrix of shape ``(n_gene, n_in)``.
        b: Per-gene bias array of shape ``(n_gene,)``.
        gamma: Scalar or per-gene linear degradation rate.
    """

    W_gene: object
    W_in: object
    b: object
    gamma: object

    def __init__(
        self,
        input_specs,
        output_specs,
        hidden_size,
        *,
        W_gene=None,
        W_in=None,
        b=None,
        gamma=0.1,
        tag='gene',
    ):
        """Build a gene circuit from explicit regulatory parameters.

        Args:
            input_specs: Per-cell driver field specifications.
            output_specs: Per-cell observed gene field specifications.
            hidden_size: Number of latent regulator genes.
            W_gene: Gene interaction matrix of shape ``(n_gene, n_gene)``. Defaults to None (zeros).
            W_in: Input mixing matrix of shape ``(n_gene, n_in)``. Defaults to None (zeros).
            b: Per-gene bias of shape ``(n_gene,)``. Defaults to None (zeros).
            gamma: Per-gene linear degradation rate, scalar or shape ``(n_gene,)``. Defaults to
                0.1.
            tag: Namespace for any latent gene field. Defaults to ``'gene'``.

        Raises:
            ValueError: If ``hidden_size`` is negative or an explicit regulatory parameter has an
                invalid shape.
        """
        super().__init__(input_specs, output_specs, hidden_size, tag=tag)
        n_gene = self.hidden_size + self.out_size
        self.W_gene = self._resolve_param(W_gene, (n_gene, n_gene), 'W_gene')
        self.W_in = self._resolve_param(W_in, (n_gene, self.in_size), 'W_in')
        self.b = self._resolve_param(b, (n_gene,), 'b')
        self.gamma = gamma

    def vector_field(self, t, evolving, inputs):
        """Return linear-drive production minus linear degradation for every cell.

        Args:
            t: Physical integration time within the macro-step, unused by this autonomous field.
            evolving: Batched latent and observed gene concentrations.
            inputs: Batched driver concentrations.

        Returns:
            Batched gene concentration derivatives.
        """
        del t
        W_gene = jnp.asarray(self.W_gene)
        W_in = jnp.asarray(self.W_in)
        b = jnp.asarray(self.b)
        gamma = jnp.asarray(self.gamma)
        drive = evolving @ W_gene.T + inputs @ W_in.T + b
        return _rescaled_sigmoid(drive) - gamma * evolving


class NeuralODE(ODEController):
    r"""Generic neural ODE controller with an MLP per-cell vector field.

    The evolving state $y$ (hidden state concatenated with outputs) obeys $\dot{y} = \mathrm{MLP}([u,
    y])$, with $u$ the fixed driver inputs over the macro-step. Build a correctly sized MLP with
    ``make_mlp``.

    Attributes:
        mlp: Equinox MLP mapping flattened inputs plus evolving state to an evolving derivative.
    """

    mlp: eqx.nn.MLP

    def __init__(
        self,
        input_specs,
        output_specs,
        hidden_size,
        *,
        mlp,
        tag='ode',
    ):
        """Build the controller from a pre-sized MLP vector field.

        Args:
            input_specs: Per-cell driver field specifications.
            output_specs: Per-cell observed evolving field specifications.
            hidden_size: Number of latent evolving values.
            mlp: An ``eqx.nn.MLP`` mapping ``in_size + hidden_size + out_size`` to
                ``hidden_size + out_size``; build one with ``make_mlp``.
            tag: Namespace for any latent state field. Defaults to ``'ode'``.

        Raises:
            ValueError: If ``hidden_size`` is negative or ``mlp`` does not have the required input
                and output sizes.
        """
        super().__init__(input_specs, output_specs, hidden_size, tag=tag)
        evolving_size = self.hidden_size + self.out_size
        expected_in = self.in_size + evolving_size
        if mlp.in_size != expected_in or mlp.out_size != evolving_size:
            raise ValueError(
                f'NeuralODE mlp must map {expected_in} -> {evolving_size} '
                f'(in_size + hidden_size + out_size -> hidden_size + out_size), '
                f'got {mlp.in_size} -> {mlp.out_size}'
            )
        self.mlp = mlp

    @classmethod
    def make_mlp(cls, input_specs, output_specs, hidden_size, *, key, width=64, depth=2):
        """Return a correctly sized ``eqx.nn.MLP`` for a ``NeuralODE`` over these fields.

        Args:
            input_specs: Per-cell driver field specifications.
            output_specs: Per-cell observed evolving field specifications.
            hidden_size: Number of latent evolving values.
            key: JAX random key used for MLP initialization.
            width: Width of every MLP hidden layer. Defaults to 64.
            depth: Number of MLP hidden layers. Defaults to 2.

        Returns:
            An MLP mapping ``in_size + hidden_size + out_size`` to ``hidden_size + out_size``.
        """
        in_size = sum(_width(spec) for spec in input_specs)
        out_size = sum(_width(spec) for spec in output_specs)
        evolving_size = hidden_size + out_size
        return eqx.nn.MLP(in_size + evolving_size, evolving_size, width, depth, key=key)

    def vector_field(self, t, evolving, inputs):
        """Return the MLP derivative for every cell's evolving controller state.

        Args:
            t: Physical integration time within the macro-step, unused by this autonomous field.
            evolving: Batched latent and observed controller state.
            inputs: Batched driver fields.

        Returns:
            Batched derivatives of the controller state.
        """
        del t
        values = jnp.concatenate([inputs, evolving], axis=1)
        return jax.vmap(self.mlp)(values)


class GeneNetworkMWC(ODEController):
    r"""Thermodynamic gene circuit with a log-occupancy (MWC-style) regulatory drive.

    Each gene concentration $g_i$ evolves as saturating production minus linear decay,

    $$\dot{g}_i = \rho_i\,\sigma(F_i) - \frac{g_i}{\tau_i},$$

    with the regulatory drive

    $$F_i = F^0_i + \sum_j H^\mathrm{gene}_{ij}\,\ln\!\left(1 + \frac{g_j}{K^\mathrm{gene}_{ij}}\right)
    + \sum_k H^\mathrm{in}_{ik}\,\ln\!\left(1 + \frac{u_k}{K^\mathrm{in}_{ik}}\right),$$

    where $u$ are the driver inputs, $\sigma$ is the sigmoid, $\rho$ the maximum production rates,
    $\tau$ the lifetimes, and $K$ the concentration thresholds. The positive quantities $\rho$, $\tau$,
    and $K$ are stored in log space and exponentiated within their dtype's finite positive range.
    Latent genes, when requested, evolve alongside the observed outputs and regulate every gene.

    Attributes:
        log_rho: Log maximum production-rate array of shape ``(n_gene,)``.
        log_tau: Log lifetime array of shape ``(n_gene,)``.
        F0: Per-gene activation-bias array of shape ``(n_gene,)``.
        H_gene: Gene-interaction array of shape ``(n_gene, n_gene)``.
        log_K_gene: Log gene-threshold array of shape ``(n_gene, n_gene)``.
        H_in: Input-mixing array of shape ``(n_gene, n_in)``.
        log_K_in: Log input-threshold array of shape ``(n_gene, n_in)``.
    """

    log_rho: object
    log_tau: object
    F0: object
    H_gene: object
    log_K_gene: object
    H_in: object
    log_K_in: object

    def __init__(
        self,
        input_specs,
        output_specs,
        *,
        hidden_size=0,
        log_rho=None,
        log_tau=None,
        F0=None,
        H_gene=None,
        log_K_gene=None,
        H_in=None,
        log_K_in=None,
        tag='gene',
    ):
        """Build the MWC circuit from explicit thermodynamic parameters.

        All parameters default to zeros, giving unit production rates and lifetimes, unit
        concentration thresholds, and an inert (zero-interaction) but well-posed circuit.

        Args:
            input_specs: Per-cell driver field specifications.
            output_specs: Per-cell observed gene field specifications.
            hidden_size: Number of latent regulator genes. Defaults to 0.
            log_rho: Log maximum production rate of shape ``(n_gene,)``. Defaults to None (zeros).
            log_tau: Log lifetime of shape ``(n_gene,)``. Defaults to None (zeros).
            F0: Per-gene activation bias of shape ``(n_gene,)``. Defaults to None (zeros).
            H_gene: Gene interaction strengths of shape ``(n_gene, n_gene)``. Defaults to None (zeros).
            log_K_gene: Log gene concentration thresholds of shape ``(n_gene, n_gene)``. Defaults
                to None (zeros).
            H_in: Input mixing strengths of shape ``(n_gene, n_in)``. Defaults to None (zeros).
            log_K_in: Log driver concentration thresholds of shape ``(n_gene, n_in)``. Defaults to
                None (zeros).
            tag: Namespace for any latent gene field. Defaults to ``'gene'``.

        Raises:
            ValueError: If ``hidden_size`` is negative or an explicit thermodynamic parameter has
                an invalid shape.
        """
        super().__init__(input_specs, output_specs, hidden_size, tag=tag)
        n_gene = self.hidden_size + self.out_size
        n_in = self.in_size
        self.log_rho = self._resolve_param(log_rho, (n_gene,), 'log_rho')
        self.log_tau = self._resolve_param(log_tau, (n_gene,), 'log_tau')
        self.F0 = self._resolve_param(F0, (n_gene,), 'F0')
        self.H_gene = self._resolve_param(H_gene, (n_gene, n_gene), 'H_gene')
        self.log_K_gene = self._resolve_param(log_K_gene, (n_gene, n_gene), 'log_K_gene')
        self.H_in = self._resolve_param(H_in, (n_gene, n_in), 'H_in')
        self.log_K_in = self._resolve_param(log_K_in, (n_gene, n_in), 'log_K_in')

    def vector_field(self, t, evolving, inputs):
        """Return thermodynamic production minus linear degradation for every cell.

        Args:
            t: Physical integration time within the macro-step, unused by this autonomous field.
            evolving: Batched latent and observed gene concentrations.
            inputs: Batched driver concentrations.

        Returns:
            Batched gene concentration derivatives.
        """
        del t
        genes = jnp.maximum(evolving, 0.0)
        drivers = jnp.maximum(inputs, 0.0)
        K_gene = _positive_from_log(jnp.asarray(self.log_K_gene))
        K_in = _positive_from_log(jnp.asarray(self.log_K_in))
        # A tiny threshold (log_K at its clip) makes the ratio overflow to +inf; clamping it keeps the
        # occupancy finite, so a mixed-sign interaction row cannot form (+inf) + (-inf) = NaN.
        max_ratio = jnp.finfo(genes.dtype).max
        gene_occupancy = jnp.log1p(jnp.minimum(genes[:, None, :] / K_gene, max_ratio))
        input_occupancy = jnp.log1p(jnp.minimum(drivers[:, None, :] / K_in, max_ratio))
        gene_drive = jnp.sum(jnp.asarray(self.H_gene) * gene_occupancy, axis=-1)
        input_drive = jnp.sum(jnp.asarray(self.H_in) * input_occupancy, axis=-1)
        rho = _positive_from_log(jnp.asarray(self.log_rho))
        tau = _positive_from_log(jnp.asarray(self.log_tau))
        production = rho * jax.nn.sigmoid(gene_drive + input_drive + jnp.asarray(self.F0))
        return production - evolving / tau
