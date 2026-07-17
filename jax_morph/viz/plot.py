"""Matplotlib-backed spatial, animation, and per-cell time-series renderers."""

from __future__ import annotations

import warnings
from functools import lru_cache
from numbers import Integral

import numpy as np

from ._prepare import (
    CellSeries,
    SpatialData,
    frame_lims,
    prepare_spatial,
    prepare_timeseries,
    resolve_3d_options,
    strip_y_lims,
    validate_alpha,
    validate_lims,
    validate_linewidth,
    validate_vlim,
)

__all__ = ['draw', 'animate', 'plot_timeseries']


def _require_matplotlib():
    """Import matplotlib only at rendering time and explain the optional dependency."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib import colormaps, colors
        from matplotlib.animation import FuncAnimation
        from matplotlib.collections import PatchCollection
        from matplotlib.patches import Circle, Patch
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except ImportError as error:
        raise ImportError(
            'jax-morph visualization requires matplotlib; install it with\n'
            '"pip install \'jax-morph[viz]\'"'
        ) from error
    return plt, colormaps, colors, FuncAnimation, PatchCollection, Circle, Patch, Poly3DCollection


def _is_3d_axes(ax) -> bool:
    """Return whether a matplotlib axes has a three-dimensional projection."""
    return getattr(ax, 'name', None) == '3d'


def _is_rectilinear_axes(ax) -> bool:
    """Return whether a matplotlib axes uses the ordinary Cartesian projection."""
    return getattr(ax, 'name', None) == 'rectilinear'


def _new_axes(plt, n_space_dim: int, ax):
    """Create or validate an axes compatible with a spatial dimensionality."""
    if ax is None:
        if n_space_dim == 3:
            fig = plt.figure()
            return fig.add_subplot(projection='3d')
        _, ax = plt.subplots()
        return ax
    compatible = _is_3d_axes(ax) if n_space_dim == 3 else _is_rectilinear_axes(ax)
    if not compatible:
        required = 'a 3-D Axes3D' if n_space_dim == 3 else 'an ordinary 2-D axes'
        raise ValueError(f'This state requires {required}.')
    return ax


def _new_timeseries_axes(plt, ax):
    """Create or validate an ordinary two-dimensional time-series axes."""
    if ax is None:
        _, ax = plt.subplots()
    elif not _is_rectilinear_axes(ax):
        raise ValueError('plot_timeseries requires an ordinary 2-D axes.')
    return ax


def _new_animation_axes(plt, n_space_dim: int, fig, ax):
    """Create or validate the one axes owned by a spatial animation."""
    if fig is not None and ax is not None:
        raise ValueError('Pass either fig or ax to animate, not both.')
    if ax is not None:
        ax = _new_axes(plt, n_space_dim, ax)
        return ax.figure, ax
    if fig is None:
        fig = plt.figure()
    elif not isinstance(fig, plt.Figure):
        raise ValueError('fig must be a matplotlib Figure.')
    elif fig.axes:
        raise ValueError('The supplied fig already contains axes; pass the intended ax instead.')
    ax = fig.add_subplot(projection='3d') if n_space_dim == 3 else fig.add_subplot()
    return fig, ax


def _finite_live_values(values: np.ndarray, alive: np.ndarray) -> np.ndarray:
    """Return finite selected values from live cells only."""
    selected = np.asarray(values)[np.asarray(alive, dtype=bool)]
    return selected[np.isfinite(selected)]


def _continuous_clim(values: np.ndarray, alive: np.ndarray) -> tuple[float, float] | None:
    """Return a finite continuous range for visible values, or None when none exist."""
    finite = _finite_live_values(values, alive)
    if finite.size == 0:
        return None
    return float(np.min(finite)), float(np.max(finite))


def _color_info(
    values: np.ndarray | None,
    alive: np.ndarray,
    *,
    categorical: bool,
    cmap: str,
    clim: tuple[float, float] | None,
    levels: np.ndarray | None,
    cm,
    colors,
):
    """Build per-cell face colors, a colorbar mappable, and categorical legend handles."""
    from matplotlib.cm import ScalarMappable

    if values is None:
        return None, None, []
    values = np.asarray(values)
    live = np.asarray(alive, dtype=bool)
    cmap_obj = cm.get_cmap(cmap)
    bad = np.asarray(cmap_obj(np.nan))
    finite = _finite_live_values(values, live)
    if finite.size == 0:
        return np.broadcast_to(bad, (*values.shape, 4)).copy(), None, []
    if categorical:
        if levels is None:
            levels = np.unique(finite)
        levels = np.asarray(levels)
        palette = cm.get_cmap(cmap).resampled(max(1, len(levels)))
        rgba = np.broadcast_to(bad, (len(values), 4)).copy()
        handles = []
        for number, level in enumerate(levels):
            color = palette(number)
            rgba[values == level] = color
            handles.append((level, color))
        return rgba, None, handles

    if clim is None:
        clim = float(np.min(finite)), float(np.max(finite))
    low, high = clim
    if low == high:
        padding = max(0.5, abs(low) * 0.01)
        norm = colors.Normalize(low - padding, high + padding, clip=False)
    else:
        norm = colors.Normalize(low, high, clip=False)
    rgba = cmap_obj(norm(np.ma.masked_where(~np.isfinite(values), values)))
    mappable = ScalarMappable(norm=norm, cmap=cmap_obj)
    mappable.set_array(np.asarray(values)[live])
    return rgba, mappable, []


def _add_categorical_legend(ax, handles, Patch) -> None:
    """Add a compact discrete legend when finite categorical levels are visible."""
    if handles:
        ax.legend(
            handles=[Patch(facecolor=color, label=str(level)) for level, color in handles],
            title='category',
        )


def _shade_sphere_faces(triangles: np.ndarray, facecolors: np.ndarray) -> np.ndarray:
    """Apply directional diffuse lighting to triangular sphere faces."""
    if len(triangles) == 0:
        return facecolors
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, lengths, out=np.zeros_like(normals), where=lengths > 0.0)
    azimuth = np.deg2rad(225.0)
    altitude = np.deg2rad(19.4712)
    light = np.array(
        [
            np.cos(altitude) * np.cos(azimuth),
            np.cos(altitude) * np.sin(azimuth),
            np.sin(altitude),
        ]
    )
    diffuse = np.clip(normals @ light, -1.0, 1.0)
    intensity = 0.65 + 0.35 * diffuse
    shaded = np.array(facecolors, copy=True)
    shaded[:, :3] = np.clip(shaded[:, :3] * intensity[:, None], 0.0, 1.0)
    return shaded


@lru_cache(maxsize=4)
def _icosphere(resolution: int) -> tuple[np.ndarray, np.ndarray]:
    """Return a cached unit icosphere topology at one bounded subdivision level."""
    golden = (1.0 + np.sqrt(5.0)) / 2.0
    vertices = np.array(
        [
            (-1, golden, 0),
            (1, golden, 0),
            (-1, -golden, 0),
            (1, -golden, 0),
            (0, -1, golden),
            (0, 1, golden),
            (0, -1, -golden),
            (0, 1, -golden),
            (golden, 0, -1),
            (golden, 0, 1),
            (-golden, 0, -1),
            (-golden, 0, 1),
        ],
        dtype=float,
    )
    faces = np.array(
        [
            (0, 11, 5),
            (0, 5, 1),
            (0, 1, 7),
            (0, 7, 10),
            (0, 10, 11),
            (1, 5, 9),
            (5, 11, 4),
            (11, 10, 2),
            (10, 7, 6),
            (7, 1, 8),
            (3, 9, 4),
            (3, 4, 2),
            (3, 2, 6),
            (3, 6, 8),
            (3, 8, 9),
            (4, 9, 5),
            (2, 4, 11),
            (6, 2, 10),
            (8, 6, 7),
            (9, 8, 1),
        ],
        dtype=int,
    )
    vertices /= np.linalg.norm(vertices, axis=1, keepdims=True)
    for _ in range(resolution):
        vertices, faces = _subdivide_icosphere(vertices, faces)
    return vertices, faces


def _subdivide_icosphere(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split every triangular icosphere face into four normalized triangles."""
    vertices_list = list(vertices)
    midpoint_index: dict[tuple[int, int], int] = {}

    def midpoint(left: int, right: int) -> int:
        key = tuple(sorted((left, right)))
        if key not in midpoint_index:
            point = vertices[left] + vertices[right]
            point /= np.linalg.norm(point)
            midpoint_index[key] = len(vertices_list)
            vertices_list.append(point)
        return midpoint_index[key]

    next_faces = []
    for left, middle, right in faces:
        a = midpoint(int(left), int(middle))
        b = midpoint(int(middle), int(right))
        c = midpoint(int(right), int(left))
        next_faces.extend(((left, a, c), (middle, b, a), (right, c, b), (a, b, c)))
    return np.asarray(vertices_list), np.asarray(next_faces, dtype=int)


def _draw_2d(
    ax,
    data: SpatialData,
    values: np.ndarray | None,
    *,
    categorical: bool,
    cmap: str,
    clim: tuple[float, float] | None,
    levels: np.ndarray | None,
    color: str,
    alpha: float,
    collections,
):
    """Draw true data-space radius circles for the live cells in one 1-D or 2-D frame."""
    _, cm, colors, _, PatchCollection, Circle, Patch, _ = collections
    live = data.alive
    centres = data.position[live]
    if data.n_space_dim == 1:
        centres = np.column_stack((centres[:, 0], np.zeros(len(centres))))
    circles = [
        Circle(center, radius) for center, radius in zip(centres, data.radius[live], strict=True)
    ]
    collection = PatchCollection(circles, alpha=alpha, edgecolor='black', linewidth=0.4)
    rgba, mappable, legend = _color_info(
        None if values is None else values[live],
        np.ones(np.count_nonzero(live), dtype=bool),
        categorical=categorical,
        cmap=cmap,
        clim=clim,
        levels=levels,
        cm=cm,
        colors=colors,
    )
    if rgba is None:
        collection.set_facecolor(color)
    else:
        collection.set_facecolor(rgba)
    ax.add_collection(collection)
    _add_categorical_legend(ax, legend, Patch)
    return mappable, collection


def _draw_3d_spheres(
    ax,
    data: SpatialData,
    values: np.ndarray | None,
    *,
    categorical: bool,
    cmap: str,
    clim: tuple[float, float] | None,
    levels: np.ndarray | None,
    color: str,
    alpha: float,
    resolution: int,
    collections,
):
    """Draw all physical-radius sphere faces in one batched Poly3DCollection."""
    _, cm, colors, _, _, _, Patch, Poly3DCollection = collections
    live = data.alive
    positions = data.position[live]
    radii = data.radius[live]
    vertices, faces = _icosphere(resolution)
    if len(positions) == 0:
        triangles = np.empty((0, 3, 3))
    else:
        transformed = positions[:, None, :] + radii[:, None, None] * vertices[None, :, :]
        offsets = np.arange(len(positions))[:, None, None] * len(vertices)
        all_faces = (faces[None, :, :] + offsets).reshape(-1, 3)
        triangles = transformed.reshape(-1, 3)[all_faces]
    rgba, mappable, legend = _color_info(
        None if values is None else values[live],
        np.ones(len(positions), dtype=bool),
        categorical=categorical,
        cmap=cmap,
        clim=clim,
        levels=levels,
        cm=cm,
        colors=colors,
    )
    if rgba is None:
        cell_colors = np.broadcast_to(colors.to_rgba(color), (len(positions), 4))
    else:
        cell_colors = rgba
    facecolors = np.repeat(cell_colors, len(faces), axis=0)
    facecolors = _shade_sphere_faces(triangles, facecolors)
    collection = Poly3DCollection(
        triangles,
        facecolors=facecolors,
        alpha=alpha,
        edgecolor='none',
        antialiased=True,
    )
    ax.add_collection3d(collection)
    _add_categorical_legend(ax, legend, Patch)
    return mappable, collection


def _draw_3d_markers(
    ax,
    data: SpatialData,
    values: np.ndarray | None,
    *,
    categorical: bool,
    cmap: str,
    clim: tuple[float, float] | None,
    levels: np.ndarray | None,
    color: str,
    alpha: float,
    marker_scale: float,
    collections,
):
    """Draw fast screen-space 3-D markers with area proportional to physical radius squared."""
    _, cm, colors, _, _, _, Patch, _ = collections
    live = data.alive
    points = data.position[live]
    sizes = marker_scale * data.radius[live] ** 2
    rgba, mappable, legend = _color_info(
        None if values is None else values[live],
        np.ones(len(points), dtype=bool),
        categorical=categorical,
        cmap=cmap,
        clim=clim,
        levels=levels,
        cm=cm,
        colors=colors,
    )
    scatter_kwargs = {
        's': sizes,
        'alpha': alpha,
        'edgecolors': 'black',
        'linewidths': 0.4,
        'depthshade': True,
    }
    if len(points) == 0:
        artist = ax.scatter([], [], [], **scatter_kwargs)
    elif rgba is None:
        artist = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=color, **scatter_kwargs)
    else:
        artist = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=rgba, **scatter_kwargs)
    _add_categorical_legend(ax, legend, Patch)
    return mappable, artist


def _set_spatial_axes(
    ax,
    data: SpatialData,
    lims: tuple[tuple[float, float], ...],
    strip_ylim: tuple[float, float] | None,
    grid: bool,
) -> None:
    """Apply physical data limits, aspect, and uncluttered axes styling."""
    ax.set_xlim(*lims[0])
    if data.n_space_dim == 1:
        ax.set_ylim(*(strip_y_lims(data.radius, data.alive) if strip_ylim is None else strip_ylim))
        ax.set_aspect('equal', adjustable='box')
        if grid:
            ax.grid(True, alpha=0.2)
        else:
            ax.grid(False)
        if not grid:
            ax.set_xticks([])
            ax.set_yticks([])
    elif data.n_space_dim == 2:
        ax.set_ylim(*lims[1])
        ax.set_aspect('equal', adjustable='box')
        if grid:
            ax.grid(True, alpha=0.2)
        else:
            ax.grid(False)
        if not grid:
            ax.set_xticks([])
            ax.set_yticks([])
    else:
        ax.set_ylim(*lims[1])
        ax.set_zlim(*lims[2])
        spans = np.array([high - low for low, high in lims])
        ax.set_box_aspect(spans)
        if grid:
            ax.grid(True, alpha=0.2)
        else:
            ax.grid(False)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_visible(grid)
            axis.line.set_visible(grid)
        if not grid:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])


def _draw_frame(
    ax,
    data: SpatialData,
    frame: int | None,
    *,
    categorical: bool,
    cmap: str,
    clim: tuple[float, float] | None,
    levels: np.ndarray | None,
    color: str,
    alpha: float,
    render_3d: str | None,
    sphere_resolution: int | None,
    marker_scale: float | None,
    lims: tuple[tuple[float, float], ...],
    strip_ylim: tuple[float, float] | None,
    grid: bool,
    collections,
):
    """Render one prepared state frame and return a numeric mappable plus its main artist."""
    if frame is None:
        position, radius, alive, values = data.position, data.radius, data.alive, data.color
    else:
        position = data.position[frame]
        radius = data.radius[frame]
        alive = data.alive[frame]
        values = None if data.color is None else data.color[frame]
    one = SpatialData(position, radius, alive, values, None, data.n_space_dim)
    if one.n_space_dim < 3:
        mappable, artist = _draw_2d(
            ax,
            one,
            values,
            categorical=categorical,
            cmap=cmap,
            clim=clim,
            levels=levels,
            color=color,
            alpha=alpha,
            collections=collections,
        )
    elif render_3d == 'spheres':
        mappable, artist = _draw_3d_spheres(
            ax,
            one,
            values,
            categorical=categorical,
            cmap=cmap,
            clim=clim,
            levels=levels,
            color=color,
            alpha=alpha,
            resolution=sphere_resolution,
            collections=collections,
        )
    else:
        mappable, artist = _draw_3d_markers(
            ax,
            one,
            values,
            categorical=categorical,
            cmap=cmap,
            clim=clim,
            levels=levels,
            color=color,
            alpha=alpha,
            marker_scale=marker_scale,
            collections=collections,
        )
    _set_spatial_axes(ax, one, lims, strip_ylim, grid)
    return mappable, artist


def _categorical_levels(values: np.ndarray | None, alive: np.ndarray) -> np.ndarray | None:
    """Compute stable finite categorical levels across a prepared state or history."""
    if values is None:
        return None
    return np.unique(_finite_live_values(values, alive))


def draw(
    state,
    *,
    color_by=None,
    cmap='viridis',
    vlim=None,
    categorical=None,
    color='tab:blue',
    colorbar=True,
    alpha=0.85,
    render_3d='spheres',
    sphere_resolution=None,
    marker_scale=None,
    ax=None,
    lims=None,
    grid=False,
    colorbar_label=None,
    title='',
):
    """Draw the live cells of one unstacked state on a native matplotlib axes.

    One-dimensional states are drawn as a circle strip, while two-dimensional circles and three-
    dimensional sphere meshes use physical data-space radii. In 3-D, ``render_3d='markers'`` is a
    faster screen-space approximation: marker area scales with ``radius**2`` but absolute size,
    overlap, and perspective are not physical. ``marker_scale`` tunes that approximation.

    Args:
        state: Unstacked simulation state to render.
        color_by: Optional scalar-per-cell field name, callable, or raw array.
        cmap: Matplotlib colormap for selected values. Defaults to ``'viridis'``.
        vlim: Optional continuous ``(low, high)`` color range. Defaults to None.
        categorical: Whether to render selected values as discrete categories. Defaults to None.
        color: Solid cell color when ``color_by`` is None. Defaults to ``'tab:blue'``.
        colorbar: Whether to add a continuous colorbar. Defaults to True.
        alpha: Cell opacity. Defaults to 0.85.
        render_3d: ``'spheres'`` for physical meshes or ``'markers'`` for fast markers.
        sphere_resolution: Icosphere subdivision level from 0 through 3. Defaults to 3 for spheres.
        marker_scale: Points-squared multiplier for 3-D markers. Defaults to 1800.0 for markers.
        ax: Optional compatible matplotlib axes. Defaults to None.
        lims: Optional physical data limits, one pair per spatial dimension. Defaults to None.
        grid: Whether to retain contextual ticks, panes, and a light grid. Defaults to False.
        colorbar_label: Optional continuous colorbar label. Named fields label themselves by default.
        title: Axes title. Defaults to an empty string.

    Returns:
        The matplotlib axes containing the cell artists.

    Note: 3-D sphere cost.
        The sphere default ``sphere_resolution=3`` favors fidelity and emits 1,280 depth-sorted
        faces per cell, so a large cluster can be slow to draw. Lower ``sphere_resolution`` or use
        ``render_3d='markers'`` for quick iteration on many cells.
    """
    collections = _require_matplotlib()
    plt = collections[0]
    data, inferred_categorical = prepare_spatial(state, color_by, history=False)
    alpha = validate_alpha(alpha)
    vlim = validate_vlim(vlim)
    render_3d, sphere_resolution, marker_scale = resolve_3d_options(
        data.n_space_dim, render_3d, sphere_resolution, marker_scale, sphere_default=3
    )
    categorical = inferred_categorical if categorical is None else bool(categorical)
    ax = _new_axes(plt, data.n_space_dim, ax)
    limits = frame_lims(
        data.position,
        data.radius,
        data.alive,
        data.n_space_dim,
        validate_lims(lims, data.n_space_dim),
    )
    clim = (
        vlim
        if vlim is not None
        else _continuous_clim(data.color, data.alive)
        if data.color is not None
        else None
    )
    levels = _categorical_levels(data.color, data.alive) if categorical else None
    mappable, _ = _draw_frame(
        ax,
        data,
        None,
        categorical=categorical,
        cmap=cmap,
        clim=clim,
        levels=levels,
        color=color,
        alpha=alpha,
        render_3d=render_3d,
        sphere_resolution=sphere_resolution,
        marker_scale=marker_scale,
        lims=limits,
        strip_ylim=None,
        grid=bool(grid),
        collections=collections,
    )
    if colorbar and mappable is not None and not categorical:
        colorbar_artist = ax.figure.colorbar(mappable, ax=ax)
        label = colorbar_label
        if label is None and isinstance(color_by, str):
            label = color_by.replace('_', ' ')
        if label:
            colorbar_artist.set_label(str(label))
    ax.set_title(title)
    return ax


def animate(
    history,
    *,
    color_by=None,
    cmap='viridis',
    vlim=None,
    categorical=None,
    color='tab:blue',
    colorbar=True,
    alpha=0.85,
    render_3d='markers',
    sphere_resolution=None,
    marker_scale=None,
    interval=300,
    fixed_lims=True,
    lims=None,
    fixed_clim=True,
    fig=None,
    ax=None,
    grid=False,
    colorbar_label=None,
    title=None,
):
    """Animate a complete stacked history or ``TrajectoryRecord`` with matplotlib.

    The function replaces only its own cell artist and categorical legend as cells appear,
    disappear, move, and change radius; other artists and axes labels remain intact. Three-
    dimensional markers are the default for predictable iteration speed; opt into
    ``render_3d='spheres'`` for physical data-space radii. The returned animation controls saving
    and notebook embedding.

    Args:
        history: Stacked complete state history or trajectory record.
        color_by: Optional scalar-per-cell field name, callable, or raw array.
        cmap: Matplotlib colormap for selected values. Defaults to ``'viridis'``.
        vlim: Optional continuous ``(low, high)`` color range. Defaults to None.
        categorical: Whether to render selected values as discrete categories. Defaults to None.
        color: Solid cell color when ``color_by`` is None. Defaults to ``'tab:blue'``.
        colorbar: Whether to add one stable continuous colorbar. Defaults to True.
        alpha: Cell opacity. Defaults to 0.85.
        render_3d: ``'spheres'`` or ``'markers'``. Defaults to ``'markers'``.
        sphere_resolution: Icosphere level from 0 through 3. Defaults to 0 for spheres.
        marker_scale: Points-squared multiplier for markers. Defaults to 1800.0.
        interval: Delay between frames in milliseconds. Defaults to 300.
        fixed_lims: Whether to keep one view across all frames. Defaults to True.
        lims: Optional persistent physical data limits, one pair per spatial dimension. Requires
            ``fixed_lims=True``. Defaults to None.
        fixed_clim: Whether to keep one continuous range across all frames. Defaults to True.
        fig: Optional empty matplotlib figure to populate. Mutually exclusive with ``ax``. Defaults
            to None.
        ax: Optional compatible matplotlib axes to populate. Mutually exclusive with ``fig``.
            Defaults to None.
        grid: Whether to retain contextual ticks, panes, and a light grid. Defaults to False.
        colorbar_label: Optional continuous colorbar label. Named fields label themselves by default.
        title: Fixed title, or None to display each frame time. Defaults to None.

    Returns:
        A matplotlib ``FuncAnimation``.
    """
    collections = _require_matplotlib()
    plt, _, _, FuncAnimation, *_ = collections
    data, inferred_categorical = prepare_spatial(history, color_by, history=True)
    alpha = validate_alpha(alpha)
    vlim = validate_vlim(vlim)
    render_3d, sphere_resolution, marker_scale = resolve_3d_options(
        data.n_space_dim, render_3d, sphere_resolution, marker_scale, sphere_default=0
    )
    categorical = inferred_categorical if categorical is None else bool(categorical)
    explicit_lims = validate_lims(lims, data.n_space_dim)
    if explicit_lims is not None and not fixed_lims:
        raise ValueError('Explicit lims require fixed_lims=True.')
    fig, ax = _new_animation_axes(plt, data.n_space_dim, fig, ax)
    n_frames = data.position.shape[0]
    all_limits = frame_lims(
        data.position.reshape(-1, data.n_space_dim),
        data.radius.reshape(-1),
        data.alive.reshape(-1),
        data.n_space_dim,
        explicit_lims,
    )
    fixed_strip_ylim = (
        strip_y_lims(data.radius.reshape(-1), data.alive.reshape(-1))
        if data.n_space_dim == 1 and fixed_lims
        else None
    )
    stable_clim = vlim
    if stable_clim is None and data.color is not None and fixed_clim and not categorical:
        stable_clim = _continuous_clim(data.color, data.alive)
    levels = _categorical_levels(data.color, data.alive) if categorical else None
    colorbar_state: dict[str, object] = {'colorbar': None}
    artist_state: dict[str, object] = {'artist': None, 'legend': None}
    resolved_colorbar_label = colorbar_label
    if resolved_colorbar_label is None and isinstance(color_by, str):
        resolved_colorbar_label = color_by.replace('_', ' ')
    if colorbar and data.color is not None and not categorical:
        initial_clim = vlim or stable_clim or _continuous_clim(data.color, data.alive)
        _, initial_mappable, _ = _color_info(
            data.color,
            data.alive,
            categorical=False,
            cmap=cmap,
            clim=initial_clim,
            levels=None,
            cm=collections[1],
            colors=collections[2],
        )
        if initial_mappable is not None:
            colorbar_state['colorbar'] = fig.colorbar(initial_mappable, ax=ax)
            if resolved_colorbar_label:
                colorbar_state['colorbar'].set_label(str(resolved_colorbar_label))

    def update(frame: int):
        previous_artist = artist_state['artist']
        if previous_artist is not None and getattr(previous_artist, 'axes', None) is ax:
            previous_artist.remove()
        previous_legend = artist_state['legend']
        if previous_legend is not None and getattr(previous_legend, 'axes', None) is ax:
            previous_legend.remove()
        limits = (
            all_limits
            if fixed_lims
            else frame_lims(
                data.position[frame], data.radius[frame], data.alive[frame], data.n_space_dim
            )
        )
        clim = stable_clim
        if data.color is not None and not categorical and not fixed_clim and vlim is None:
            clim = _continuous_clim(data.color[frame], data.alive[frame])
        mappable, artist = _draw_frame(
            ax,
            data,
            frame,
            categorical=categorical,
            cmap=cmap,
            clim=clim,
            levels=levels,
            color=color,
            alpha=alpha,
            render_3d=render_3d,
            sphere_resolution=sphere_resolution,
            marker_scale=marker_scale,
            lims=limits,
            strip_ylim=fixed_strip_ylim,
            grid=bool(grid),
            collections=collections,
        )
        artist_state['artist'] = artist
        artist_state['legend'] = (
            ax.get_legend()
            if categorical
            and data.color is not None
            and _finite_live_values(data.color[frame], data.alive[frame]).size
            else None
        )
        current_colorbar = colorbar_state['colorbar']
        if colorbar and not categorical and mappable is not None:
            if current_colorbar is None:
                colorbar_state['colorbar'] = fig.colorbar(mappable, ax=ax)
                if resolved_colorbar_label:
                    colorbar_state['colorbar'].set_label(str(resolved_colorbar_label))
            else:
                current_colorbar.update_normal(mappable)
                current_colorbar.ax.set_visible(True)
        elif current_colorbar is not None:
            current_colorbar.ax.set_visible(False)
        ax.set_title(f't = {data.t[frame]:g}' if title is None else title)
        return [artist]

    return FuncAnimation(fig, update, frames=n_frames, interval=interval, blit=False)


def _select_cell_ids(
    cell_ids, records: tuple[CellSeries, ...], known_ids: tuple[int, ...]
) -> list[int]:
    """Validate public cell-id filtering while retaining caller order."""
    record_ids = {record.cell_id for record in records}
    known = set(known_ids)
    if cell_ids is None:
        selected = sorted(record_ids)
        if len(selected) > 10:
            warnings.warn(
                f'plot_timeseries is drawing {len(selected)} cell traces; pass cell_ids=... to '
                'select a manageable subset.',
                UserWarning,
                stacklevel=3,
            )
        return selected
    if isinstance(cell_ids, Integral) and not isinstance(cell_ids, bool):
        selected = [int(cell_ids)]
    else:
        try:
            selected = list(cell_ids)
        except TypeError as error:
            raise ValueError('cell_ids must be an integer or iterable of integers.') from error
    if any(isinstance(cell_id, bool) or not isinstance(cell_id, Integral) for cell_id in selected):
        raise ValueError('cell_ids must contain non-boolean integers.')
    selected = [int(cell_id) for cell_id in selected]
    if any(cell_id < 0 for cell_id in selected):
        raise ValueError('cell_ids must be non-negative.')
    if len(set(selected)) != len(selected):
        raise ValueError('cell_ids must not contain duplicates.')
    unknown = [cell_id for cell_id in selected if cell_id not in known]
    if unknown:
        raise ValueError(f'Unknown cell ids: {unknown}.')
    return selected


def plot_timeseries(
    history,
    field,
    *,
    index=None,
    cell_ids=None,
    ax=None,
    alpha=0.85,
    linewidth=1.5,
    title='',
    ylabel=None,
):
    """Plot one selected cell-scope value against recorded simulation time.

    Cell traces use trajectory-local reconstructed ids rather than storage slots, so a slot reused
    after death creates a new line. Named fields and callables may have trailing components selected
    by ``index``; raw arrays are already scalar per cell and may be constant or frame-aligned.

    Args:
        history: Stacked complete state history or trajectory record.
        field: Cell field name, callable selector, or scalar raw array.
        index: Optional complete trailing component index for a named or callable field.
        cell_ids: Optional cell id or ordered iterable of ids to draw.
        ax: Optional ordinary matplotlib axes. Defaults to None.
        alpha: Line opacity. Defaults to 0.85.
        linewidth: Non-negative matplotlib line width. Defaults to 1.5.
        title: Axes title. Defaults to an empty string.
        ylabel: Optional y-axis label overriding the selected-field default.

    Returns:
        The matplotlib axes containing one line per selected cell identity.
    """
    collections = _require_matplotlib()
    plt = collections[0]
    alpha = validate_alpha(alpha)
    linewidth = validate_linewidth(linewidth)
    t, _, values, records, known_ids, default_ylabel = prepare_timeseries(history, field, index)
    selected = _select_cell_ids(cell_ids, records, known_ids)
    ax = _new_timeseries_axes(plt, ax)
    by_id = {record.cell_id: record for record in records}
    for cell_id in selected:
        record = by_id.get(cell_id)
        if record is None:
            continue
        x = t[record.start : record.stop]
        y = values[record.start : record.stop, record.slot]
        if np.issubdtype(y.dtype, np.bool_):
            y = y.astype(float)
        else:
            y = np.where(np.isfinite(y), y, np.nan)
        kwargs = {'label': f'cell {cell_id}', 'alpha': alpha, 'linewidth': linewidth}
        if len(x) == 1:
            kwargs['marker'] = 'o'
        ax.plot(x, y, **kwargs)
    ax.set_xlabel('simulation time')
    ax.set_ylabel(default_ylabel if ylabel is None else ylabel)
    ax.set_title(title)
    return ax
