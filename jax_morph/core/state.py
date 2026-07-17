"""Typed simulation state, synthesized from the model's declared fields.

A model's state is a generated ``BaseState`` subclass whose fields (the base fields plus every
field the model's steps require) are real ``jax.Array`` attributes - so ``state.position`` and
``state.custom_field`` both work, no dict indexing. ``build_state_from_model(model, name)`` builds
(and caches) that class. It has two constructors: the plain field-wise
``StateCls(**field_arrays, space=...)`` (pass every field directly), and
``StateCls.init_empty(capacity=..., n_space_dim=..., n_types=...)`` which allocates every field to
its default (an empty, zero-live-cell state). Either way, setting an initial condition is the
simulation's job (``state.update(...)``, or a small seeding helper it defines for itself).

Fields have a ``scope``: ``'cell'`` (per-cell - leading capacity axis, alive-masked, resized on
division) or ``'global'`` (not indexed by cell: simulation time ``t``, a scalar, or an Eulerian grid
array). Global scope is storage only - it carries the array so a custom layer can add time-dependent
forcing or global feedback; genuine grid coupling (interpolation, deposition, stencils, boundary
conditions) is not provided and remains substantial custom work.

The always-present base fields have prefilled specs (``POSITION``, ``RADIUS``, ``CELLTYPE``,
``ALIVE``, ``TIME``) that a step can use directly as an input/output, or call to override an
attribute (e.g. ``POSITION(heritable=False)``). The generated state constructor owns the concrete
shapes of the dimension-dependent base fields (``position`` -> ``(n_space_dim,)``, ``celltype`` ->
``(n_types,)``).
"""

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace

import equinox as eqx
import jax
import jax.numpy as jnp

from .geometry import free_space

__all__ = [
    'StateFieldSpec',
    'BaseState',
    'POSITION',
    'RADIUS',
    'CELLTYPE',
    'ALIVE',
    'TIME',
    'BASE_SPECS',
    'BASE_FIELDS',
    'BASE_GLOBALS',
    'merge_specs',
    'build_state_from_model',
]


@dataclass(frozen=True)
class StateFieldSpec:
    """Declaration of one state field: its name, how to allocate it, and its division behaviour.

    A spec is self-describing - it carries its own ``name`` - so steps declare inputs/outputs as
    plain tuples of specs. Calling a spec returns a copy with the given attributes overridden, so a
    prefilled base spec can be customised inline, e.g. ``POSITION(heritable=False)``.

    Attributes:
        name: The field's name; becomes a real attribute on the generated state class.
        shape: Trailing array shape. A cell-scope field has full shape
            ``(capacity, *shape)``; a global field has shape ``shape``. Defaults to ``()``.
        dtype: Array dtype. Defaults to None (the JAX default float).
        heritable: Whether a daughter cell inherits the mother's value on division (``'cell'``
            scope only). ``True`` marks a heritable property of the cell itself (size, type,
            internal chemical state): the daughter is born with a copy of the mother's value.
            ``False`` marks a transient or contextual quantity (a recorded action, a per-step
            signal): the daughter starts at ``default`` instead. Defaults to True.
        default: Fill value used to allocate empty states, to reset ephemeral trace fields each
            macro-step, and to initialise non-inherited (``heritable=False``) fields of newborn
            daughters. Defaults to 0.0.
        scope: ``'cell'`` (per-cell: leading capacity axis, alive-masked, resized on division) or
            ``'global'`` (not indexed by cell: simulation time, scalars, grids). Defaults to
            ``'cell'``.
    """

    name: str
    shape: tuple = ()
    dtype: object = None
    heritable: bool = True
    default: float = 0.0
    scope: str = 'cell'

    def __call__(self, **overrides: object) -> 'StateFieldSpec':
        """Return a copy of this specification with selected attributes overridden.

        Args:
            **overrides: Field attributes to replace.

        Returns:
            A new specification containing the requested replacements.

        Raises:
            TypeError: If an override does not name a specification field.
        """
        return replace(self, **overrides)


# Prefilled specs for the always-present base fields; usable as step inputs/outputs like any other.
# The state constructor owns the shapes of the dimension-dependent ones (position, celltype).
POSITION = StateFieldSpec('position')  # shape (n_space_dim,), filled by the constructor
RADIUS = StateFieldSpec('radius')
CELLTYPE = StateFieldSpec('celltype')  # shape (n_types,), filled by the constructor
ALIVE = StateFieldSpec('alive', dtype=bool, default=False)
TIME = StateFieldSpec('t', scope='global', default=0.0)  # simulation time

BASE_SPECS = (POSITION, RADIUS, CELLTYPE, ALIVE, TIME)
# The constructor resolves these trailing shapes for the base fields (others default to ()).
_BASE_SHAPES = {
    'position': lambda n_space_dim, n_types: (n_space_dim,),
    'celltype': lambda n_space_dim, n_types: (n_types,),
}
BASE_FIELDS = tuple(spec.name for spec in BASE_SPECS if spec.scope == 'cell')
BASE_GLOBALS = tuple(spec.name for spec in BASE_SPECS if spec.scope == 'global')


def merge_specs(*spec_iterables: Iterable[StateFieldSpec]) -> dict[str, StateFieldSpec]:
    """Merge iterables of field specs into a ``{name: spec}`` dict (keyed by each spec's name).

    Deduplicates specs that are exactly equal; raises when two specs share a ``name`` but differ.

    Args:
        *spec_iterables: Iterables of `StateFieldSpec` objects.

    Returns:
        Specifications keyed by field name, preserving first-seen order.

    Raises:
        ValueError: If two declarations for one name differ.
    """
    out: dict[str, StateFieldSpec] = {}
    for specs in spec_iterables:
        for spec in specs:
            if spec.name in out and out[spec.name] != spec:
                raise ValueError(
                    f'Field {spec.name!r} declared with conflicting specs: '
                    f'{out[spec.name]} vs {spec}'
                )
            out[spec.name] = spec
    return out


class BaseState(eqx.Module):
    """Base class for a model's state; generated subclasses add the fields as real attributes.

    Carries the static schema (``specs``) and space (``displacement``/``shift``) and the shared
    field-access helpers. Instances are always of a ``build_state_from_model`` subclass whose
    per-cell / global fields are real ``jax.Array`` attributes (``state.position``,
    ``state.custom_field``), so this base declares none of them itself.

    Attributes:
        specs: Static mapping from field names to `StateFieldSpec` declarations. Generated
            subclasses add one array attribute per entry, with shape ``(capacity, *spec.shape)``
            for cell-scope fields and ``spec.shape`` for global fields.
        displacement: Static binary function mapping two positions of shape ``(dim,)`` to a
            displacement of shape ``(dim,)``.
        shift: Static binary function applying a displacement of shape ``(dim,)`` to a position
            of shape ``(dim,)``.

    Methods:
        set: Functionally replace one field.
        update: Functionally replace several fields.
        deltas: Build a sparse dynamic-step update.
    """

    specs: Mapping = eqx.field(static=True)
    displacement: Callable = eqx.field(static=True)
    shift: Callable = eqx.field(static=True)

    def __getitem__(self, name: str) -> jax.Array:
        """Return the field array named ``name``."""
        return getattr(self, name)

    def set(self, name: str, value: jax.Array) -> 'BaseState':
        """Return a copy with one field replaced.

        Args:
            name: Name of the state field to replace.
            value: Replacement array.

        Returns:
            A same-typed state containing the replacement.
        """
        return eqx.tree_at(lambda s: getattr(s, name), self, value)

    def update(self, **kw: jax.Array) -> 'BaseState':
        """Return a copy with several fields replaced by keyword.

        Args:
            **kw: Field names and replacement arrays.

        Returns:
            A same-typed state containing all replacements.
        """
        names = list(kw)
        return eqx.tree_at(lambda s: [getattr(s, n) for n in names], self, list(kw.values()))

    def deltas(self, **fields: jax.Array) -> 'BaseState':
        """Return a sparse delta state: the named fields set, every other field ``None``.

        The result is a same-typed pytree that carries this state's static schema and space but
        holds an array only for the fields passed in (each a per-``dt`` increment) and ``None``
        everywhere else - the Equinox "updates" idiom. A dynamic step returns one of these; the
        model sums them across dynamic writers and applies them with ``eqx.apply_updates``, so
        untouched fields (``None``) are left exactly as they were and bool/discrete base fields like
        ``alive`` are never accidentally integrated.

        Args:
            **fields: Per-``dt`` increments keyed by field name.

        Returns:
            A same-typed sparse state with ``None`` in every untouched array leaf.
        """
        empty = jax.tree_util.tree_map(lambda _: None, self)
        if not fields:
            return empty
        names = list(fields)
        return eqx.tree_at(
            lambda s: [getattr(s, n) for n in names],
            empty,
            list(fields.values()),
            is_leaf=lambda x: x is None,
        )

    @property
    def n_cells(self):
        """Number of cell slots (the allocated capacity)."""
        return self.position.shape[0]

    @property
    def n_space_dim(self):
        """Spatial dimension."""
        return self.position.shape[1]


_STATE_CLASS_CACHE: dict = {}


def _make_state_cls(cls_name, specs):
    """Generate (memoised) a ``BaseState`` subclass with one ``jax.Array`` field per spec.

    The subclass has a plain field-wise constructor (pass every field as an array) plus an
    ``init_empty`` classmethod that allocates an empty state; it is cached by name + specs so a
    given model always yields the same class (no JAX recompilation / pytree-structure churn).
    """
    cache_key = (cls_name, tuple(sorted(specs.items(), key=lambda kv: kv[0])))
    cached = _STATE_CLASS_CACHE.get(cache_key)
    if cached is not None:
        return cached

    annotations = {name: jax.Array for name in specs}

    def __init__(
        self,
        *,
        space: tuple[Callable, Callable] | None = None,
        **fields: jax.Array,
    ) -> None:
        """Construct a state directly from its field arrays (the plain field-wise constructor).

        Every schema field must be given as a keyword array - base and custom alike; ``space`` is
        the optional boundary metric (default free space). The static schema (``specs``) is filled
        automatically. Use ``init_empty`` to allocate an empty state instead of passing arrays.

        Args:
            self: Generated state instance being initialized.
            space: ``(displacement, shift)`` space pair. Defaults to None (free space).
            **fields: One array for every declared state field.

        Raises:
            TypeError: If a schema field is missing or an unknown field is supplied.
        """
        missing = [name for name in specs if name not in fields]
        unexpected = [name for name in fields if name not in specs]
        if missing or unexpected:
            raise TypeError(f'{cls_name}: missing fields {missing}, unexpected fields {unexpected}')
        disp, shift = free_space() if space is None else space
        self.specs = specs
        self.displacement = disp
        self.shift = shift
        for name in specs:
            setattr(self, name, fields[name])

    def init_empty(
        cls,
        *,
        capacity: int,
        n_space_dim: int,
        n_types: int,
        space: tuple[Callable, Callable] | None = None,
    ) -> 'BaseState':
        """Allocate an empty (zero-live-cell) state: every field at its ``spec.default``.

        Base and custom fields are allocated identically - per-cell fields get a leading
        ``capacity`` axis, globals (like time ``t``) do not. Set an initial condition afterwards
        with ``state.update(...)``.

        Args:
            cls: Generated state class to instantiate.
            capacity: Number of allocated cell slots.
            n_space_dim: Spatial dimension used for ``position``.
            n_types: Cell-type vector width used for ``celltype``.
            space: ``(displacement, shift)`` space pair. Defaults to None (free space).

        Returns:
            An empty instance of the generated state class.
        """
        fields = {}
        for name, spec in specs.items():
            trailing = (
                _BASE_SHAPES[name](n_space_dim, n_types) if name in _BASE_SHAPES else spec.shape
            )
            shape = trailing if spec.scope == 'global' else (capacity, *trailing)
            fields[name] = (
                jnp.full(shape, spec.default, dtype=float)  # dtype None -> JAX default float
                if spec.dtype is None
                else jnp.full(shape, spec.default, dtype=spec.dtype)
            )
        return cls(space=space, **fields)

    cls = type(
        cls_name,
        (BaseState,),
        {
            '__annotations__': annotations,
            '__init__': __init__,
            'init_empty': classmethod(init_empty),
        },
    )
    _STATE_CLASS_CACHE[cache_key] = cls
    return cls


def build_state_from_model(model: object, name: str | None = None) -> type[BaseState]:
    """Generate (and cache) a ``BaseState`` subclass for ``model`` with fields as real attributes.

    The full schema is the always-present base fields (the state's) merged with the model's
    ``state_requires()``. The returned class has two constructors: the plain field-wise
    ``StateCls(**field_arrays, space=...)`` and ``StateCls.init_empty(capacity=..., n_space_dim=...,
    n_types=..., space=...)`` (allocate an empty state); seed an initial condition with
    ``state.update(...)``. ``name`` sets the class name and is capitalized (``'elongation'`` ->
    ``ElongationState``); default is ``State``.

    Args:
        model: Model whose declared field requirements define the schema.
        name: Optional generated class name or prefix. Defaults to None (the class is named ``State``).

    Returns:
        A cached `BaseState` subclass with one array attribute per required field.

    Raises:
        ValueError: If model steps declare conflicting specifications for one field.
    """
    specs = merge_specs(BASE_SPECS, model.state_requires())
    if name is None:
        cls_name = 'State'
    else:
        name = name[0].upper() + name[1:]  # enforce a capitalized class name
        cls_name = name if name.endswith('State') else f'{name}State'
    return _make_state_cls(cls_name, specs)
