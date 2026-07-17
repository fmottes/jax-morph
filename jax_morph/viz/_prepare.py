"""Validation and host-side data preparation shared by visualization backends."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from numbers import Integral

import jax
import jax.numpy as jnp
import numpy as np

from jax_morph.core.serialization import TrajectoryRecord
from jax_morph.core.state import BaseState


@dataclass(frozen=True)
class SpatialData:
    """Host arrays needed to render one state or a complete spatial history."""

    position: np.ndarray
    radius: np.ndarray
    alive: np.ndarray
    color: np.ndarray | None
    t: np.ndarray | None
    n_space_dim: int


@dataclass(frozen=True)
class CellSeries:
    """One private storage interval representing a trajectory-local cell identity."""

    cell_id: int
    slot: int
    start: int
    stop: int


def _is_history(state: BaseState) -> bool:
    """Return whether ``state`` has the leading frame axis used by complete histories."""
    return state.position.ndim == 3


def _is_real_numeric_or_bool(dtype: object) -> bool:
    """Return whether ``dtype`` can be rendered as a real scalar quantity."""
    dtype = np.dtype(dtype)
    return bool(
        np.issubdtype(dtype, np.bool_)
        or (np.issubdtype(dtype, np.number) and not np.issubdtype(dtype, np.complexfloating))
    )


def _require_real_numeric_or_bool(values: jax.Array, name: str) -> None:
    """Raise a clear error when a selected renderer value cannot be plotted."""
    if not _is_real_numeric_or_bool(values.dtype):
        raise ValueError(f'{name} must have a real numeric or boolean dtype, got {values.dtype}.')


def _as_jax_array(value: object, name: str) -> jax.Array:
    """Convert a caller array to JAX while keeping conversion failures user-facing."""
    try:
        return jnp.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f'{name} must be an array-like value.') from error


def _require_state(state: object) -> BaseState:
    """Return a state after enforcing the public visualization state contract."""
    if not isinstance(state, BaseState):
        raise ValueError('Visualization expects a jax_morph BaseState.')
    return state


def _validate_common_state(state: BaseState, *, history: bool) -> tuple[int, int | None]:
    """Validate state ranks and declared capacity axes; return capacity and frame count."""
    state = _require_state(state)
    if history:
        if state.position.ndim != 3:
            raise ValueError('Expected a stacked complete history, not an unstacked state.')
        n_frames, capacity, n_space_dim = state.position.shape
        if n_frames < 1:
            raise ValueError('A history must contain at least one frame.')
        if state.radius.ndim != 2 or state.radius.shape != (n_frames, capacity):
            raise ValueError('History radius must have shape (n_frames, capacity).')
        if state.alive.ndim != 2 or state.alive.shape != (n_frames, capacity):
            raise ValueError('History alive must have shape (n_frames, capacity).')
        if state.t.ndim != 1 or state.t.shape != (n_frames,):
            raise ValueError('History t must have shape (n_frames,).')
    else:
        if state.position.ndim != 2:
            raise ValueError('Expected an unstacked state, not a stacked history.')
        capacity, n_space_dim = state.position.shape
        n_frames = None
        if state.radius.ndim != 1 or state.radius.shape != (capacity,):
            raise ValueError('State radius must have shape (capacity,).')
        if state.alive.ndim != 1 or state.alive.shape != (capacity,):
            raise ValueError('State alive must have shape (capacity,).')
        if state.t.ndim != 0:
            raise ValueError('State t must be a scalar.')

    for name, spec in state.specs.items():
        value = getattr(state, name)
        if spec.scope == 'cell':
            expected = (n_frames, capacity) if history else (capacity,)
            if value.ndim < len(expected) or tuple(value.shape[: len(expected)]) != expected:
                raise ValueError(
                    f'Cell field {name!r} does not have the required '
                    f'{"frame/capacity" if history else "capacity"} axes.'
                )
        elif history and (value.ndim < 1 or value.shape[0] != n_frames):
            raise ValueError(f'Global history field {name!r} is not frame-aligned.')

    if n_space_dim < 1 or n_space_dim > 3:
        raise ValueError('Visualization supports spatial dimensions from 1 through 3.')
    return capacity, n_frames


def _validate_time(t: jax.Array) -> None:
    """Validate the dtype of recorded simulation times before their batched host transfer."""
    if not _is_real_numeric_or_bool(t.dtype) or np.issubdtype(np.dtype(t.dtype), np.bool_):
        raise ValueError('History t must have a real numeric dtype.')


def _normalize_history(history: object) -> BaseState:
    """Unwrap a trajectory record and validate the stacked-history shape contract."""
    if isinstance(history, TrajectoryRecord):
        history = history.history
    history = _require_state(history)
    _validate_common_state(history, history=True)
    _validate_time(history.t)
    return history


def _resolve_cell_field(
    state: BaseState, selector: str | Callable[[BaseState], jax.Array]
) -> jax.Array:
    """Resolve a named or callable cell field while it remains on the JAX device.

    The returned array retains all trailing field components.  A stacked history maps callable
    selectors over frames, so those selectors must be JAX-transformable.
    """
    history = _is_history(state)
    capacity, n_frames = _validate_common_state(state, history=history)
    if isinstance(selector, str):
        if selector not in state.specs:
            raise ValueError(f'Unknown state field {selector!r}.')
        if state.specs[selector].scope != 'cell':
            raise ValueError(f'Field {selector!r} is global and cannot select per-cell values.')
        values = getattr(state, selector)
    elif callable(selector):
        try:
            values = jax.vmap(selector)(state) if history else selector(state)
        except Exception as error:  # JAX reports shape/tracing details in the chained exception.
            raise ValueError('Callable cell selector must be JAX-transformable.') from error
        values = _as_jax_array(values, 'Callable cell selector result')
    else:
        raise ValueError('A cell field selector must be a field name or callable.')

    expected = (n_frames, capacity) if history else (capacity,)
    if values.ndim < len(expected) or tuple(values.shape[: len(expected)]) != expected:
        raise ValueError(
            'Cell selector output must begin with '
            f'{"(n_frames, capacity)" if history else "(capacity,)"} axes.'
        )
    return values


def _resolve_spatial_color(
    state: BaseState,
    color_by: object,
    *,
    history: bool,
) -> tuple[jax.Array | None, bool]:
    """Resolve and validate a scalar-per-cell spatial color selector."""
    capacity, n_frames = _validate_common_state(state, history=history)
    expected = (n_frames, capacity) if history else (capacity,)
    if color_by is None:
        return None, False

    inferred_categorical = color_by == 'celltype' if isinstance(color_by, str) else False
    if inferred_categorical:
        values = jnp.argmax(state.celltype, axis=-1)
    elif isinstance(color_by, str) or callable(color_by):
        values = _resolve_cell_field(state, color_by)
    else:
        values = _as_jax_array(color_by, 'color_by')
        allowed = [expected]
        if history:
            allowed.append((capacity,))
        if tuple(values.shape) not in allowed:
            forms = '(capacity,) or (n_frames, capacity)' if history else '(capacity,)'
            raise ValueError(f'Raw color_by must have shape {forms}.')
        if history and values.ndim == 1:
            values = jnp.broadcast_to(values, expected)

    if tuple(values.shape) == (*expected, 1):
        values = values[..., 0]
    if tuple(values.shape) != expected:
        raise ValueError(
            'Spatial color_by must resolve to one scalar per cell; vector-valued fields are not '
            'valid colors.'
        )
    _require_real_numeric_or_bool(values, 'color_by')
    return values, inferred_categorical


def _validate_live_geometry(position: np.ndarray, radius: np.ndarray, alive: np.ndarray) -> None:
    """Reject invalid geometry only where a cell is alive."""
    live = np.asarray(alive, dtype=bool)
    if not np.all(np.isfinite(position[live])):
        raise ValueError('Live cell positions must be finite.')
    live_radius = radius[live]
    if not np.all(np.isfinite(live_radius)):
        raise ValueError('Live cell radii must be finite.')
    if np.any(live_radius < 0):
        raise ValueError('Live cell radii must be non-negative.')


def prepare_spatial(
    state: BaseState,
    color_by: object,
    *,
    history: bool,
) -> tuple[SpatialData, bool]:
    """Resolve one spatial dataset and transfer only its renderer inputs to the host."""
    if history:
        state = _normalize_history(state)
    else:
        _validate_common_state(state, history=False)
    values, inferred_categorical = _resolve_spatial_color(state, color_by, history=history)
    position, radius, alive, color, t = jax.device_get(
        (state.position, state.radius, state.alive, values, state.t if history else None)
    )
    position = np.asarray(position)
    radius = np.asarray(radius)
    alive = np.asarray(alive, dtype=bool)
    color = None if color is None else np.asarray(color)
    t = None if t is None else np.asarray(t)
    if t is not None and not np.all(np.isfinite(t)):
        raise ValueError('History t must contain only finite values.')
    _validate_live_geometry(position, radius, alive)
    return SpatialData(
        position, radius, alive, color, t, state.position.shape[-1]
    ), inferred_categorical


def validate_alpha(alpha: object) -> float:
    """Return a valid renderer alpha value."""
    try:
        alpha = float(alpha)
    except (TypeError, ValueError) as error:
        raise ValueError('alpha must be a finite number in [0, 1].') from error
    if not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError('alpha must be a finite number in [0, 1].')
    return alpha


def validate_linewidth(linewidth: object) -> float:
    """Return a valid non-negative line width."""
    try:
        linewidth = float(linewidth)
    except (TypeError, ValueError) as error:
        raise ValueError('linewidth must be a finite non-negative number.') from error
    if not np.isfinite(linewidth) or linewidth < 0:
        raise ValueError('linewidth must be a finite non-negative number.')
    return linewidth


def validate_vlim(vlim: object) -> tuple[float, float] | None:
    """Validate a continuous color range."""
    if vlim is None:
        return None
    try:
        low, high = vlim
        low, high = float(low), float(high)
    except (TypeError, ValueError) as error:
        raise ValueError('vlim must contain two finite values with low < high.') from error
    if not np.isfinite(low) or not np.isfinite(high) or not low < high:
        raise ValueError('vlim must contain two finite values with low < high.')
    return low, high


def validate_lims(lims: object, n_space_dim: int) -> tuple[tuple[float, float], ...] | None:
    """Validate caller-provided displayed spatial limits."""
    if lims is None:
        return None
    try:
        values = tuple(tuple(pair) for pair in lims)
    except TypeError as error:
        raise ValueError(
            'lims must contain one finite (low, high) pair per spatial dimension.'
        ) from error
    if len(values) != n_space_dim:
        raise ValueError('lims must contain one finite (low, high) pair per spatial dimension.')
    out = []
    for pair in values:
        if len(pair) != 2:
            raise ValueError('lims must contain one finite (low, high) pair per spatial dimension.')
        low, high = map(float, pair)
        if not np.isfinite(low) or not np.isfinite(high) or not low < high:
            raise ValueError('lims must contain two finite values with low < high.')
        out.append((low, high))
    return tuple(out)


def resolve_3d_options(
    n_space_dim: int,
    render_3d: object,
    sphere_resolution: object,
    marker_scale: object,
    *,
    sphere_default: int,
) -> tuple[str | None, int | None, float | None]:
    """Validate mode-specific arguments and resolve the documented 3-D defaults."""
    if render_3d not in ('spheres', 'markers'):
        raise ValueError("render_3d must be 'spheres' or 'markers'.")
    if n_space_dim != 3:
        if sphere_resolution is not None or marker_scale is not None:
            raise ValueError('sphere_resolution and marker_scale are only valid for 3-D rendering.')
        return None, None, None
    if render_3d == 'spheres':
        if marker_scale is not None:
            raise ValueError("marker_scale is only valid when render_3d='markers'.")
        resolution = sphere_default if sphere_resolution is None else sphere_resolution
        if isinstance(resolution, bool) or not isinstance(resolution, Integral):
            raise ValueError('sphere_resolution must be a non-boolean integer from 0 through 3.')
        resolution = int(resolution)
        if not 0 <= resolution <= 3:
            raise ValueError('sphere_resolution must be a non-boolean integer from 0 through 3.')
        return 'spheres', resolution, None

    if sphere_resolution is not None:
        raise ValueError("sphere_resolution is only valid when render_3d='spheres'.")
    scale = 1800.0 if marker_scale is None else marker_scale
    try:
        scale = float(scale)
    except (TypeError, ValueError) as error:
        raise ValueError('marker_scale must be a finite positive number.') from error
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('marker_scale must be a finite positive number.')
    return 'markers', None, scale


def frame_lims(
    position: np.ndarray,
    radius: np.ndarray,
    alive: np.ndarray,
    n_space_dim: int,
    lims: tuple[tuple[float, float], ...] | None = None,
) -> tuple[tuple[float, float], ...]:
    """Return physical displayed limits from live centres plus radii or explicit limits."""
    if lims is not None:
        return lims
    live = np.asarray(alive, dtype=bool)
    if not np.any(live):
        return tuple((-1.0, 1.0) for _ in range(n_space_dim))
    pos = position[live]
    rad = radius[live]
    out = []
    for axis in range(n_space_dim):
        low = float(np.min(pos[:, axis] - rad))
        high = float(np.max(pos[:, axis] + rad))
        if low == high:
            pad = max(1.0, abs(low) * 0.1)
            low, high = low - pad, high + pad
        out.append((low, high))
    return tuple(out)


def strip_y_lims(radius: np.ndarray, alive: np.ndarray) -> tuple[float, float]:
    """Return the non-physical vertical extent used for a one-dimensional circle strip."""
    live_radius = np.asarray(radius)[np.asarray(alive, dtype=bool)]
    extent = 1.0 if live_radius.size == 0 else max(1.0, float(np.max(live_radius)) * 1.5)
    return -extent, extent


def _index_values(values: jax.Array, index: object) -> jax.Array:
    """Apply a complete trailing component index and require one scalar per history cell."""
    trailing = values.ndim - 2
    if index is None:
        if trailing == 0:
            return values
        if trailing == 1 and values.shape[-1] == 1:
            return values[..., 0]
        raise ValueError('A vector-valued field needs index selecting every trailing component.')

    if isinstance(index, bool):
        raise ValueError('index must be an integer or a tuple of integers.')
    indices = (index,) if isinstance(index, Integral) else index
    if not isinstance(indices, tuple) or len(indices) != trailing:
        raise ValueError('index must contain exactly one integer for each trailing field axis.')
    normalized = []
    for component, size in zip(indices, values.shape[2:], strict=True):
        if isinstance(component, bool) or not isinstance(component, Integral):
            raise ValueError('index must be an integer or a tuple of integers.')
        component = int(component)
        if not -size <= component < size:
            raise ValueError('index is out of bounds for a trailing field axis.')
        normalized.append(component)
    result = values[(slice(None), slice(None), *normalized)]
    if result.ndim != 2:
        raise ValueError('index must select one scalar value per cell.')
    return result


def prepare_timeseries(
    history: object,
    field: object,
    index: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[CellSeries, ...], tuple[int, ...], str]:
    """Resolve one scalar cell value and reconstruct private trajectory-local cell intervals."""
    history = _normalize_history(history)
    n_frames, capacity = history.alive.shape
    if isinstance(field, str) or callable(field):
        values = _resolve_cell_field(history, field)
        ylabel = field if isinstance(field, str) else 'value'
    else:
        values = _as_jax_array(field, 'field')
        if tuple(values.shape) == (capacity,):
            values = jnp.broadcast_to(values, (n_frames, capacity))
        elif tuple(values.shape) != (n_frames, capacity):
            raise ValueError(
                'Raw time-series field must have shape (capacity,) or (n_frames, capacity).'
            )
        ylabel = 'value'
    values = _index_values(values, index)
    _require_real_numeric_or_bool(values, 'field')
    if isinstance(field, str) and index is not None:
        index_text = str(index) if isinstance(index, Integral) else str(tuple(index))
        ylabel = f'{field}[{index_text}]'

    has_lineage = 'born' in history.specs and 'mother' in history.specs
    born = history.born if has_lineage else None
    t, alive, values, born = jax.device_get((history.t, history.alive, values, born))
    t = np.asarray(t)
    alive = np.asarray(alive, dtype=bool)
    values = np.asarray(values)
    born = None if born is None else np.asarray(born)
    if not np.all(np.isfinite(t)):
        raise ValueError('History t must contain only finite values.')
    records, known_ids = reconstruct_cell_series(alive, born)
    return t, alive, values, records, known_ids, ylabel


def reconstruct_cell_series(
    alive: np.ndarray,
    born: np.ndarray | None,
) -> tuple[tuple[CellSeries, ...], tuple[int, ...]]:
    """Reconstruct private cell-id storage intervals without exposing slots to callers."""
    alive = np.asarray(alive, dtype=bool)
    n_frames, capacity = alive.shape
    frame_ids = np.full((n_frames, capacity), -1, dtype=int)
    next_id = 0
    if born is not None:
        born = np.asarray(born)
        if born.shape != alive.shape:
            raise ValueError('History born must have shape (n_frames, capacity).')
        slot_to_id: dict[int, int] = {}
        for slot in np.flatnonzero(alive[0]):
            slot_to_id[int(slot)] = next_id
            frame_ids[0, slot] = next_id
            next_id += 1
        for frame in range(1, n_frames):
            for slot in np.flatnonzero(born[frame] > 0.5):
                slot_to_id[int(slot)] = next_id
                next_id += 1
            for slot in np.flatnonzero(alive[frame]):
                slot = int(slot)
                if slot not in slot_to_id:
                    # A malformed lineage history can still be visualized as its observable runs.
                    slot_to_id[slot] = next_id
                    next_id += 1
                frame_ids[frame, slot] = slot_to_id[slot]
            for slot in tuple(slot_to_id):
                if not alive[frame, slot]:
                    del slot_to_id[slot]
    else:
        current = np.full(capacity, -1, dtype=int)
        for frame in range(n_frames):
            starts = alive[frame] & ((frame == 0) | ~alive[frame - 1])
            for slot in np.flatnonzero(starts):
                current[slot] = next_id
                next_id += 1
            frame_ids[frame, alive[frame]] = current[alive[frame]]
            current[~alive[frame]] = -1

    records = []
    for cell_id in range(next_id):
        frames, slots = np.where(frame_ids == cell_id)
        if frames.size:
            records.append(
                CellSeries(cell_id, int(slots[0]), int(frames.min()), int(frames.max()) + 1)
            )
    return tuple(records), tuple(range(next_id))
