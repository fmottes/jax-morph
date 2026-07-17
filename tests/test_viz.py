import builtins
import warnings

import matplotlib

matplotlib.use('Agg')

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytest

import jax_morph as jxm
from jax_morph.viz import _prepare
from jax_morph.viz import plot as viz_plot


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close('all')


class _VizModel:
    def state_requires(self):
        return (
            jxm.StateFieldSpec('value'),
            jxm.StateFieldSpec('vector', shape=(2,)),
            jxm.StateFieldSpec('global_value', scope='global'),
            jxm.StateFieldSpec('born', heritable=False),
            jxm.StateFieldSpec('mother', dtype=int, default=-1, heritable=False),
        )


class _NoLineageModel:
    def state_requires(self):
        return (jxm.StateFieldSpec('value'),)


class _SingletonFieldModel:
    def state_requires(self):
        return (jxm.StateFieldSpec('singleton', shape=(1,)),)


def _state(n_space_dim=2):
    State = jxm.build_state_from_model(_VizModel())
    state = State.init_empty(capacity=3, n_space_dim=n_space_dim, n_types=2)
    positions = jnp.zeros((3, n_space_dim)).at[1, 0].set(2.0)
    return state.update(
        alive=jnp.array([True, True, False]),
        position=positions,
        radius=jnp.array([0.5, 1.0, 0.0]),
        celltype=jnp.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]),
        value=jnp.array([1.0, 2.0, 0.0]),
        vector=jnp.array([[1.0, 10.0], [2.0, 20.0], [0.0, 0.0]]),
    )


def _history(state, n_frames=2):
    return jax.tree_util.tree_map(
        lambda value: jnp.repeat(value[None], n_frames, axis=0), state
    ).update(t=jnp.linspace(0.0, 2.5, n_frames))


def _record(history):
    return jxm.TrajectoryRecord(history=history, dt=jnp.asarray(0.25), provenance={})


def _render_animation(animation):
    animation._draw_was_started = True
    return [animation._func(frame) for frame in animation.new_frame_seq()]


def _no_lineage_history():
    State = jxm.build_state_from_model(_NoLineageModel())
    state = State.init_empty(capacity=2, n_space_dim=2, n_types=1).update(
        radius=jnp.ones(2),
        celltype=jnp.ones((2, 1)),
        value=jnp.zeros(2),
    )
    return _history(state, n_frames=4).update(
        t=jnp.array([0.0, 1.0, 2.0, 3.0]),
        alive=jnp.array([[True, False], [False, False], [True, False], [True, False]]),
        value=jnp.array([[1.0, 0.0], [2.0, 0.0], [10.0, 0.0], [11.0, 0.0]]),
    )


def test_draw_uses_true_2d_circle_radii_and_named_colours():
    ax = jxm.viz.draw(_state(), color_by='value')

    collection = ax.collections[0]
    assert len(collection.get_paths()) == 2
    vertices = collection.get_paths()[0].vertices
    assert np.isclose(vertices[:, 0].max() - vertices[:, 0].min(), 1.0)
    assert len(ax.figure.axes) == 2


def test_draw_3d_markers_scale_area_from_physical_radius():
    state = _state(3)

    markers = jxm.viz.draw(state, render_3d='markers', marker_scale=25.0).collections[0]
    assert np.allclose(markers.get_sizes(), [6.25, 25.0])
    assert np.allclose(markers.get_linewidths(), [0.4])
    assert np.allclose(markers.get_edgecolors()[0, :3], 0.0)


def test_draw_3d_spheres_have_smooth_lit_faces():
    ax = jxm.viz.draw(_state(3), color='tab:blue', sphere_resolution=2)
    facecolors = ax.collections[0].get_facecolors()

    assert len(facecolors) == 2 * 320
    assert len(np.unique(np.round(facecolors[:, :3], 4), axis=0)) > 20


def test_draw_3d_grid_switches_between_clean_and_contextual_axes():
    clean = jxm.viz.draw(_state(3), grid=False)
    contextual = jxm.viz.draw(_state(3), grid=True)

    assert not clean.xaxis.pane.get_visible()
    assert not clean.yaxis.pane.get_visible()
    assert not clean.zaxis.pane.get_visible()
    assert len(clean.get_xticks()) == 0
    assert contextual.xaxis.pane.get_visible()
    assert contextual.yaxis.pane.get_visible()
    assert contextual.zaxis.pane.get_visible()
    assert len(contextual.get_xticks()) > 0


def test_draw_labels_continuous_colorbars_from_named_fields_with_override():
    automatic = jxm.viz.draw(_state(), color_by='value')
    overridden = jxm.viz.draw(_state(), color_by='value', colorbar_label='Signal level')

    assert automatic.figure.axes[1].get_ylabel() == 'value'
    assert overridden.figure.axes[1].get_ylabel() == 'Signal level'


def test_draw_rejects_global_or_vector_colours_and_inapplicable_3d_knobs():
    state = _state()
    with pytest.raises(ValueError, match='global'):
        jxm.viz.draw(state, color_by='global_value')
    with pytest.raises(ValueError, match='scalar'):
        jxm.viz.draw(state, color_by='vector')
    with pytest.raises(ValueError, match='3-D'):
        jxm.viz.draw(state, sphere_resolution=0)


def test_draw_1d_uses_physical_radii_and_one_explicit_spatial_limit():
    ax = jxm.viz.draw(_state(1), lims=((-5.0, 5.0),))

    paths = ax.collections[0].get_paths()
    assert len(paths) == 2
    assert np.isclose(np.ptp(paths[0].vertices[:, 0]), 1.0)
    assert np.allclose(ax.get_xlim(), (-5.0, 5.0))
    assert np.allclose(ax.get_ylim(), (-1.5, 1.5))


@pytest.mark.parametrize('resolution', range(4))
def test_icosphere_geometry_has_the_documented_face_count_and_unit_radius(resolution):
    vertices, faces = viz_plot._icosphere(resolution)

    assert faces.shape == (20 * 4**resolution, 3)
    assert np.allclose(np.linalg.norm(vertices, axis=1), 1.0)


@pytest.mark.parametrize('resolution', range(4))
def test_draw_3d_batches_transformed_physical_sphere_faces(monkeypatch, resolution):
    collections = viz_plot._require_matplotlib()
    real_collection = collections[-1]
    captured = {}

    class _CapturingCollection(real_collection):
        def __init__(self, vertices, *args, **kwargs):
            captured['triangles'] = np.asarray(vertices)
            super().__init__(vertices, *args, **kwargs)

    monkeypatch.setattr(
        viz_plot, '_require_matplotlib', lambda: (*collections[:-1], _CapturingCollection)
    )
    state = _state(3)
    ax = jxm.viz.draw(state, sphere_resolution=resolution)

    faces_per_cell = 20 * 4**resolution
    triangles = captured['triangles']
    assert triangles.shape == (2 * faces_per_cell, 3, 3)
    assert len(ax.collections) == 1
    assert np.allclose(
        np.linalg.norm(triangles[:faces_per_cell] - np.asarray(state.position[0]), axis=-1),
        state.radius[0],
    )
    assert np.allclose(
        np.linalg.norm(triangles[faces_per_cell:] - np.asarray(state.position[1]), axis=-1),
        state.radius[1],
    )


def test_draw_3d_box_aspect_tracks_unequal_displayed_spans():
    ax = jxm.viz.draw(
        _state(3),
        lims=((0.0, 2.0), (-1.0, 3.0), (-2.0, 6.0)),
        render_3d='markers',
    )
    aspect = np.asarray(ax.get_box_aspect())

    assert np.allclose(aspect / aspect[0], [1.0, 2.0, 4.0])


@pytest.mark.parametrize(
    ('kwargs', 'message'),
    [
        ({'render_3d': 'invalid'}, 'spheres.*markers'),
        ({'sphere_resolution': True}, 'non-boolean integer'),
        ({'sphere_resolution': -1}, '0 through 3'),
        ({'sphere_resolution': 4}, '0 through 3'),
        ({'sphere_resolution': 1.5}, 'non-boolean integer'),
        ({'render_3d': 'spheres', 'marker_scale': 10.0}, 'marker_scale'),
        ({'render_3d': 'markers', 'sphere_resolution': 0}, 'sphere_resolution'),
        ({'render_3d': 'markers', 'marker_scale': 0.0}, 'positive'),
        ({'render_3d': 'markers', 'marker_scale': np.inf}, 'positive'),
    ],
)
def test_draw_rejects_invalid_3d_modes_and_parameters(kwargs, message):
    with pytest.raises(ValueError, match=message):
        jxm.viz.draw(_state(3), **kwargs)


def test_draw_validates_projection_in_both_directions():
    figure = plt.figure()
    ordinary = figure.add_subplot(121)
    axes_3d = figure.add_subplot(122, projection='3d')

    with pytest.raises(ValueError, match='3-D Axes3D'):
        jxm.viz.draw(_state(3), ax=ordinary)
    with pytest.raises(ValueError, match='ordinary 2-D axes'):
        jxm.viz.draw(_state(), ax=axes_3d)


def test_draw_rejects_unsupported_dimensions():
    with pytest.raises(ValueError, match='dimensions from 1 through 3'):
        jxm.viz.draw(_state(4))


def test_draw_masks_dead_geometry_and_accepts_negative_live_positions():
    state = _state().update(
        position=jax.device_put(np.array([[-3.0, -2.0], [2.0, 0.0], [np.nan, np.inf]])),
        radius=jax.device_put(np.array([0.5, 1.0, np.nan])),
    )

    ax = jxm.viz.draw(state)

    assert ax.get_xlim()[0] <= -3.5
    assert len(ax.collections[0].get_paths()) == 2


@pytest.mark.parametrize(
    ('updates', 'message'),
    [
        (
            {'position': jax.device_put(np.array([[np.nan, 0.0], [2.0, 0.0], [0.0, 0.0]]))},
            'positions',
        ),
        ({'radius': jax.device_put(np.array([np.inf, 1.0, 0.0]))}, 'radii'),
        ({'radius': jnp.array([-0.1, 1.0, 0.0])}, 'non-negative'),
    ],
)
def test_draw_rejects_invalid_live_geometry(updates, message):
    with pytest.raises(ValueError, match=message):
        jxm.viz.draw(_state().update(**updates))


def test_draw_empty_state_has_default_limits_and_no_cells():
    state = _state().update(alive=jnp.zeros(3, dtype=bool))

    ax = jxm.viz.draw(state)

    assert np.allclose(ax.get_xlim(), (-1.0, 1.0))
    assert np.allclose(ax.get_ylim(), (-1.0, 1.0))
    assert len(ax.collections[0].get_paths()) == 0


@pytest.mark.parametrize(
    'color_by',
    [
        pytest.param(lambda state: state.vector[:, 0], id='callable'),
        pytest.param(np.array([1.0, 2.0, 3.0]), id='raw'),
    ],
)
def test_draw_accepts_callable_and_raw_scalar_colors(color_by):
    ax = jxm.viz.draw(_state(), color_by=color_by)

    assert len(ax.figure.axes) == 2


def test_draw_squeezes_a_singleton_cell_field():
    State = jxm.build_state_from_model(_SingletonFieldModel())
    state = State.init_empty(capacity=2, n_space_dim=2, n_types=1).update(
        alive=jnp.ones(2, dtype=bool),
        radius=jnp.ones(2),
        celltype=jnp.ones((2, 1)),
        singleton=jnp.array([[1.0], [2.0]]),
    )

    ax = jxm.viz.draw(state, color_by='singleton')

    assert len(ax.figure.axes) == 2


def test_draw_celltype_is_categorical_with_stable_discrete_labels():
    ax = jxm.viz.draw(_state(), color_by='celltype')

    assert len(ax.figure.axes) == 1
    assert [text.get_text() for text in ax.get_legend().get_texts()] == ['0', '1']


def test_draw_nonfinite_colors_use_bad_color_and_all_nonfinite_omits_colorbar():
    values = np.array([np.inf, 1.0, 0.0])
    ax = jxm.viz.draw(_state(), color_by=values)
    expected_bad = matplotlib.colormaps['viridis'].get_bad()

    assert np.allclose(ax.collections[0].get_facecolors()[0, :3], expected_bad[:3])
    assert np.isclose(ax.collections[0].get_facecolors()[0, 3], 0.85)

    all_bad = jxm.viz.draw(_state(), color_by=jax.device_put(np.full(3, np.nan)))
    assert len(all_bad.figure.axes) == 1
    assert len(all_bad.collections[0].get_paths()) == 2


def test_draw_constant_color_uses_colormap_midpoint():
    ax = jxm.viz.draw(_state(), color_by=np.ones(3))
    midpoint = matplotlib.colormaps['viridis'](0.5)

    assert np.allclose(ax.collections[0].get_facecolors()[0, :3], midpoint[:3])
    assert np.isclose(ax.collections[0].get_facecolors()[0, 3], 0.85)


@pytest.mark.parametrize(
    ('kwargs', 'message'),
    [
        ({'color_by': 'missing'}, 'Unknown'),
        ({'color_by': np.ones((3, 2))}, 'shape'),
        ({'color_by': np.ones(2)}, 'shape'),
        ({'color_by': np.ones(3, dtype=complex)}, 'real numeric'),
        ({'vlim': (1.0, 1.0)}, 'low < high'),
        ({'vlim': (0.0, np.inf)}, 'finite'),
        ({'lims': ((0.0, 1.0),)}, 'one finite'),
        ({'lims': ((1.0, 0.0), (0.0, 1.0))}, 'low < high'),
        ({'alpha': -0.1}, 'alpha'),
        ({'alpha': np.nan}, 'alpha'),
    ],
)
def test_draw_rejects_invalid_selectors_and_display_ranges(kwargs, message):
    with pytest.raises(ValueError, match=message):
        jxm.viz.draw(_state(), **kwargs)


def test_draw_rejects_history_and_animate_rejects_state():
    with pytest.raises(ValueError, match='unstacked'):
        jxm.viz.draw(_history(_state()))
    with pytest.raises(ValueError, match='stacked complete history'):
        jxm.viz.animate(_state())


def test_animate_keeps_one_colourbar_axes_through_updates():
    animation = jxm.viz.animate(_history(_state()), color_by='value')
    animation._draw_was_started = True
    animation._func(0)
    assert len(animation._fig.axes) == 2
    animation._func(1)
    assert len(animation._fig.axes) == 2


def test_animate_reuses_supplied_axes_and_preserves_custom_artists_and_labels():
    figure, supplied = plt.subplots()
    reference = supplied.axhline(0.5, color='red')
    supplied.set_xlabel('x coordinate')
    supplied.set_ylabel('y coordinate')

    animation = jxm.viz.animate(_history(_state()), ax=supplied, colorbar=False, grid=True)
    _render_animation(animation)

    assert animation._fig is figure
    assert len(figure.axes) == 1
    assert reference in supplied.lines
    assert supplied.get_xlabel() == 'x coordinate'
    assert supplied.get_ylabel() == 'y coordinate'
    assert len(supplied.collections) == 1


def test_animate_rejects_ambiguous_or_nonempty_figure_targets():
    figure, supplied = plt.subplots()

    with pytest.raises(ValueError, match='already contains axes'):
        jxm.viz.animate(_history(_state()), fig=figure)
    with pytest.raises(ValueError, match='either fig or ax'):
        jxm.viz.animate(_history(_state()), fig=figure, ax=supplied)


def test_animate_validates_supplied_axes_projection_in_both_directions():
    figure = plt.figure()
    ordinary = figure.add_subplot(121)
    axes_3d = figure.add_subplot(122, projection='3d')

    with pytest.raises(ValueError, match='3-D Axes3D'):
        jxm.viz.animate(_history(_state(3)), ax=ordinary)
    with pytest.raises(ValueError, match='ordinary 2-D axes'):
        jxm.viz.animate(_history(_state()), ax=axes_3d)


def test_animate_accepts_explicit_persistent_limits_and_rejects_dynamic_conflict():
    history = _history(_state()).update(
        position=jnp.array(
            [
                [[0.0, 0.0], [2.0, 0.0], [0.0, 0.0]],
                [[10.0, 0.0], [14.0, 0.0], [0.0, 0.0]],
            ]
        )
    )
    limits = ((-5.0, 5.0), (-3.0, 3.0))
    animation = jxm.viz.animate(history, lims=limits, colorbar=False)
    animation._draw_was_started = True

    animation._func(0)
    assert np.allclose(animation._fig.axes[0].get_xlim(), limits[0])
    assert np.allclose(animation._fig.axes[0].get_ylim(), limits[1])
    animation._func(1)
    assert np.allclose(animation._fig.axes[0].get_xlim(), limits[0])
    assert np.allclose(animation._fig.axes[0].get_ylim(), limits[1])

    with pytest.raises(ValueError, match='fixed_lims'):
        jxm.viz.animate(history, lims=limits, fixed_lims=False)


def test_animate_retains_context_grid_and_custom_colorbar_label():
    animation = jxm.viz.animate(
        _history(_state(3)),
        color_by='value',
        grid=True,
        colorbar_label='Signal level',
    )
    animation._draw_was_started = True

    animation._func(0)

    assert len(animation._fig.axes[0].get_xticks()) > 0
    assert animation._fig.axes[1].get_ylabel() == 'Signal level'


def test_animate_1d_fixed_limits_include_the_visual_strip_axis():
    history = _history(_state(1)).update(radius=jnp.array([[0.5, 1.0, 0.0], [4.0, 1.0, 0.0]]))

    animation = jxm.viz.animate(history, fixed_lims=True)
    animation._draw_was_started = True
    animation._func(0)
    first_ylim = animation._fig.axes[0].get_ylim()
    animation._func(1)

    assert animation._fig.axes[0].get_ylim() == first_ylim


def test_spatial_and_timeseries_axes_reject_non_rectilinear_projections():
    figure = plt.figure()
    polar = figure.add_subplot(projection='polar')
    with pytest.raises(ValueError, match='ordinary 2-D axes'):
        jxm.viz.draw(_state(), ax=polar)
    with pytest.raises(ValueError, match='ordinary 2-D axes'):
        jxm.viz.plot_timeseries(_history(_state()), 'value', ax=polar)


def test_categorical_legend_uses_a_selector_neutral_title():
    ax = jxm.viz.draw(_state(), color_by='value', categorical=True, colorbar=False)

    assert ax.get_legend().get_title().get_text() == 'category'


def test_animate_accepts_trajectory_record_and_updates_every_frame_with_recorded_time():
    history = _history(_state(), n_frames=4).update(t=jnp.array([0.0, 0.5, 3.0, 8.0]))
    animation = jxm.viz.animate(_record(history), color_by='value')

    results = _render_animation(animation)

    assert len(results) == 4
    assert all(len(result) == 1 for result in results)
    assert len(animation._fig.axes) == 2
    assert len(animation._fig.axes[0].collections) == 1
    assert animation._fig.axes[0].get_title() == 't = 8'


def test_animate_fixed_and_dynamic_spatial_limits():
    history = _history(_state()).update(
        position=jnp.array(
            [
                [[0.0, 0.0], [2.0, 0.0], [0.0, 0.0]],
                [[10.0, 0.0], [14.0, 0.0], [0.0, 0.0]],
            ]
        )
    )
    fixed = jxm.viz.animate(history, fixed_lims=True, colorbar=False)
    dynamic = jxm.viz.animate(history, fixed_lims=False, colorbar=False)
    fixed._draw_was_started = True
    dynamic._draw_was_started = True

    fixed._func(0)
    fixed_first = fixed._fig.axes[0].get_xlim()
    fixed._func(1)
    dynamic._func(0)
    dynamic_first = dynamic._fig.axes[0].get_xlim()
    dynamic._func(1)

    assert fixed._fig.axes[0].get_xlim() == fixed_first
    assert dynamic._fig.axes[0].get_xlim() != dynamic_first


def test_animate_fixed_and_dynamic_continuous_color_ranges():
    history = _history(_state()).update(value=jnp.array([[0.0, 1.0, 0.0], [100.0, 200.0, 0.0]]))
    fixed = jxm.viz.animate(history, color_by='value', colorbar=False, fixed_clim=True)
    dynamic = jxm.viz.animate(history, color_by='value', colorbar=False, fixed_clim=False)
    fixed._draw_was_started = True
    dynamic._draw_was_started = True

    fixed_first = fixed._func(0)[0].get_facecolors().copy()
    fixed_second = fixed._func(1)[0].get_facecolors().copy()
    dynamic_first = dynamic._func(0)[0].get_facecolors().copy()
    dynamic_second = dynamic._func(1)[0].get_facecolors().copy()

    assert not np.allclose(fixed_first[0], fixed_second[0])
    assert np.allclose(dynamic_first[0], dynamic_second[0])


def test_animate_categorical_colors_are_stable_when_a_level_disappears():
    history = _history(_state()).update(value=jnp.array([[0.0, 1.0, 0.0], [1.0, 2.0, 0.0]]))
    animation = jxm.viz.animate(
        history, color_by='value', categorical=True, colorbar=False, fixed_clim=False
    )
    animation._draw_was_started = True

    first = animation._func(0)[0].get_facecolors().copy()
    second = animation._func(1)[0].get_facecolors().copy()

    assert np.allclose(first[1], second[0])
    assert animation._fig.axes[0].get_legend().get_title().get_text() == 'category'


@pytest.mark.parametrize(
    'color_by',
    [
        pytest.param(np.array([1.0, 2.0, 3.0]), id='constant-raw'),
        pytest.param(np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]), id='per-frame-raw'),
    ],
)
def test_animate_accepts_constant_and_per_frame_raw_colors(color_by):
    animation = jxm.viz.animate(_history(_state()), color_by=color_by)

    _render_animation(animation)

    assert len(animation._fig.axes) == 2


def test_animate_handles_empty_frames_and_all_empty_histories():
    history = _history(_state()).update(
        alive=jnp.array([[False, False, False], [True, False, False]])
    )
    animation = jxm.viz.animate(history)
    animation._draw_was_started = True

    empty_artist = animation._func(0)[0]
    live_artist = animation._func(1)[0]

    assert len(empty_artist.get_paths()) == 0
    assert len(live_artist.get_paths()) == 1

    all_empty = jxm.viz.animate(
        history.update(alive=jnp.zeros((2, 3), dtype=bool)), fixed_lims=True
    )
    _render_animation(all_empty)
    assert np.allclose(all_empty._fig.axes[0].get_xlim(), (-1.0, 1.0))


def test_draw_and_animate_resolve_documented_sphere_defaults(monkeypatch):
    calls = []
    original = viz_plot._icosphere

    def recording_icosphere(resolution):
        calls.append(resolution)
        return original(resolution)

    monkeypatch.setattr(viz_plot, '_icosphere', recording_icosphere)
    jxm.viz.draw(_state(3))
    animation = jxm.viz.animate(_history(_state(3)), render_3d='spheres')
    animation._draw_was_started = True
    animation._func(0)

    assert calls == [3, 0]
    assert len(animation._fig.axes[0].collections) == 1


def test_draw_and_animate_3d_markers_use_readable_default_scale_and_interval():
    drawn = jxm.viz.draw(_state(3), render_3d='markers')
    animation = jxm.viz.animate(_history(_state(3)))
    animation._draw_was_started = True

    artist = animation._func(0)[0]

    assert np.allclose(drawn.collections[0].get_sizes(), [450.0, 1800.0])
    assert np.allclose(artist.get_sizes(), [450.0, 1800.0])
    assert animation.event_source.interval == 300


def test_animate_rejects_nonfinite_recorded_time():
    history = _history(_state()).update(t=jax.device_put(np.array([0.0, np.nan])))

    with pytest.raises(ValueError, match='finite'):
        jxm.viz.animate(history)


def test_timeseries_uses_times_and_never_connects_a_reused_slot():
    state = _state()
    history = _history(state, n_frames=4).update(
        t=jnp.array([0.0, 1.0, 4.0, 7.0]),
        alive=jnp.array(
            [
                [True, False, False],
                [False, False, False],
                [True, False, False],
                [True, False, False],
            ]
        ),
        born=jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        mother=jnp.full((4, 3), -1, dtype=int),
        value=jnp.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [10.0, 0.0, 0.0], [11.0, 0.0, 0.0]]),
        vector=jnp.zeros((4, 3, 2)),
    )
    ax = jxm.viz.plot_timeseries(history, 'value')

    assert [line.get_label() for line in ax.lines] == ['cell 0', 'cell 1']
    assert np.array_equal(ax.lines[0].get_xdata(), [0.0])
    assert np.array_equal(ax.lines[1].get_xdata(), [4.0, 7.0])
    assert ax.get_xlabel() == 'simulation time'


def test_timeseries_indexes_vector_fields_and_warns_for_many_default_traces():
    history = _history(_state())
    ax = jxm.viz.plot_timeseries(history, 'vector', index=1)
    assert ax.get_ylabel() == 'vector[1]'
    assert np.array_equal(ax.lines[0].get_ydata(), [10.0, 10.0])

    State = jxm.build_state_from_model(_VizModel())
    crowded = State.init_empty(capacity=11, n_space_dim=2, n_types=2).update(
        alive=jnp.ones(11, dtype=bool),
        radius=jnp.ones(11),
        celltype=jnp.ones((11, 2)),
        value=jnp.ones(11),
        vector=jnp.ones((11, 2)),
    )
    crowded = _history(crowded)
    with pytest.warns(UserWarning, match='11'):
        jxm.viz.plot_timeseries(crowded, 'value')


def test_timeseries_accepts_record_callable_and_raw_values_with_labels_and_overrides():
    history = _history(_state()).update(t=jnp.array([2.0, 9.0]))
    figure, supplied = plt.subplots()
    supplied.axhline(3.0)

    callable_ax = jxm.viz.plot_timeseries(
        _record(history),
        lambda state: state.vector[:, 0] * 2,
        ax=supplied,
        title='selected values',
        ylabel='double value',
    )
    raw_ax = jxm.viz.plot_timeseries(history, np.array([4.0, 5.0, 6.0]), cell_ids=[1])

    assert callable_ax is supplied
    assert len(callable_ax.lines) == 3
    assert np.array_equal(callable_ax.lines[1].get_xdata(), [2.0, 9.0])
    assert np.array_equal(callable_ax.lines[1].get_ydata(), [2.0, 2.0])
    assert callable_ax.get_ylabel() == 'double value'
    assert callable_ax.get_title() == 'selected values'
    assert raw_ax.get_ylabel() == 'value'
    assert np.array_equal(raw_ax.lines[0].get_ydata(), [5.0, 5.0])


def test_timeseries_cell_id_filter_preserves_order_and_accepts_empty_selection():
    history = _history(_state())

    ordered = jxm.viz.plot_timeseries(history, 'value', cell_ids=[1, 0])
    empty = jxm.viz.plot_timeseries(history, 'value', cell_ids=[])

    assert [line.get_label() for line in ordered.lines] == ['cell 1', 'cell 0']
    assert len(empty.lines) == 0


@pytest.mark.parametrize(
    ('cell_ids', 'message'),
    [
        (True, 'integer or iterable'),
        ([False], 'non-boolean integers'),
        ([-1], 'non-negative'),
        ([0, 0], 'duplicates'),
        ([99], 'Unknown'),
        ([0.5], 'non-boolean integers'),
    ],
)
def test_timeseries_rejects_invalid_cell_id_filters(cell_ids, message):
    with pytest.raises(ValueError, match=message):
        jxm.viz.plot_timeseries(_history(_state()), 'value', cell_ids=cell_ids)


@pytest.mark.parametrize('capacity', [10, 11])
def test_timeseries_warning_threshold_and_explicit_selection(capacity):
    State = jxm.build_state_from_model(_VizModel())
    state = State.init_empty(capacity=capacity, n_space_dim=2, n_types=1).update(
        alive=jnp.ones(capacity, dtype=bool),
        radius=jnp.ones(capacity),
        celltype=jnp.ones((capacity, 1)),
        value=jnp.ones(capacity),
        vector=jnp.ones((capacity, 2)),
    )
    history = _history(state)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        jxm.viz.plot_timeseries(history, 'value')
    assert len(caught) == (1 if capacity == 11 else 0)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        jxm.viz.plot_timeseries(history, 'value', cell_ids=list(range(capacity)))
    assert caught == []


def test_timeseries_fallback_identity_splits_alive_runs_and_raw_values():
    history = _no_lineage_history()

    field_ax = jxm.viz.plot_timeseries(history, 'value')
    raw_ax = jxm.viz.plot_timeseries(history, np.asarray(history.value))

    assert [line.get_label() for line in field_ax.lines] == ['cell 0', 'cell 1']
    assert np.array_equal(field_ax.lines[0].get_ydata(), [1.0])
    assert np.array_equal(field_ax.lines[1].get_ydata(), [10.0, 11.0])
    assert np.array_equal(raw_ax.lines[0].get_xdata(), [0.0])
    assert np.array_equal(raw_ax.lines[1].get_xdata(), [2.0, 3.0])


def test_timeseries_lineage_ids_match_reconstruction_order():
    history = _history(_state(), n_frames=3).update(
        t=jnp.array([0.0, 1.0, 2.0]),
        alive=jnp.array([[True, False, False], [True, True, False], [True, True, True]]),
        born=jnp.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        mother=jnp.array([[-1, -1, -1], [-1, 0, -1], [-1, -1, 0]]),
        value=jnp.array([[1.0, 0.0, 0.0], [2.0, 10.0, 0.0], [3.0, 11.0, 20.0]]),
        vector=jnp.zeros((3, 3, 2)),
    )
    nodes = jxm.physics.reconstruct_lineage(history.born, history.mother, history.alive)

    ax = jxm.viz.plot_timeseries(history, 'value')

    assert [line.get_label() for line in ax.lines] == [f'cell {node["id"]}' for node in nodes]
    assert [line.get_xdata()[0] for line in ax.lines] == [0.0, 1.0, 2.0]


def test_timeseries_accepts_known_cell_born_and_killed_between_frames():
    history = _history(_state()).update(
        alive=jnp.array([[True, False, False], [True, False, False]]),
        born=jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        mother=jnp.array([[-1, -1, -1], [-1, -1, 0]]),
    )

    ax = jxm.viz.plot_timeseries(history, 'value', cell_ids=[1])

    assert len(ax.lines) == 0


def test_timeseries_celltype_component_is_distinct_from_argmax_callable():
    history = _history(_state())

    component = jxm.viz.plot_timeseries(history, 'celltype', index=0)
    active_type = jxm.viz.plot_timeseries(history, lambda state: state.celltype.argmax(-1))

    assert np.array_equal(component.lines[1].get_ydata(), [0.0, 0.0])
    assert np.array_equal(active_type.lines[1].get_ydata(), [1, 1])


def test_timeseries_one_frame_cells_are_visible_and_empty_history_has_no_lines():
    history = _no_lineage_history()

    one_frame = jxm.viz.plot_timeseries(history, 'value', cell_ids=[0])
    empty = jxm.viz.plot_timeseries(history.update(alive=jnp.zeros((4, 2), dtype=bool)), 'value')

    assert one_frame.lines[0].get_marker() == 'o'
    assert len(empty.lines) == 0


def test_timeseries_replaces_nonfinite_values_with_gaps():
    history = _history(_state(), n_frames=3).update(
        t=jnp.array([0.0, 1.0, 2.0]),
        value=jax.device_put(np.array([[1.0, 0.0, 0.0], [np.inf, 0.0, 0.0], [3.0, 0.0, 0.0]])),
        vector=jnp.zeros((3, 3, 2)),
    )

    ax = jxm.viz.plot_timeseries(history, 'value', cell_ids=[0])

    assert np.array_equal(ax.lines[0].get_xdata(), [0.0, 1.0, 2.0])
    assert np.isnan(ax.lines[0].get_ydata()[1])


@pytest.mark.parametrize(
    ('field', 'index', 'message'),
    [
        ('global_value', None, 'global'),
        ('vector', None, 'needs index'),
        ('vector', True, 'index'),
        ('vector', (0, 1), 'exactly one'),
        ('vector', 2, 'out of bounds'),
        (np.ones((2, 3, 1)), None, 'shape'),
        (np.ones((2, 3)), 0, 'exactly one'),
        (np.ones((2, 3), dtype=complex), None, 'real numeric'),
        (np.array(['a', 'b', 'c']), None, 'array-like'),
    ],
)
def test_timeseries_rejects_invalid_fields_and_indices(field, index, message):
    with pytest.raises(ValueError, match=message):
        jxm.viz.plot_timeseries(_history(_state()), field, index=index)


def test_timeseries_rejects_unstacked_state_and_bad_callable_shape():
    with pytest.raises(ValueError, match='stacked complete history'):
        jxm.viz.plot_timeseries(_state(), 'value')
    with pytest.raises(ValueError, match='begin with'):
        jxm.viz.plot_timeseries(_history(_state()), lambda state: state.value[:2])


def test_prepare_transfers_each_renderer_dataset_once(monkeypatch):
    real_device_get = _prepare.jax.device_get
    calls = []

    def recording_device_get(value):
        calls.append(value)
        return real_device_get(value)

    monkeypatch.setattr(_prepare.jax, 'device_get', recording_device_get)

    jxm.viz.draw(_state(), color_by='value')
    assert len(calls) == 1

    calls.clear()
    jxm.viz.plot_timeseries(_history(_state()), 'value')
    assert len(calls) == 1


def test_backend_import_error_names_the_visualization_extra(monkeypatch):
    real_import = builtins.__import__

    def import_without_matplotlib(name, *args, **kwargs):
        if name == 'matplotlib.pyplot':
            raise ImportError('matplotlib is unavailable')
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', import_without_matplotlib)

    with pytest.raises(ImportError, match=r'jax-morph\[viz\]'):
        viz_plot._require_matplotlib()
