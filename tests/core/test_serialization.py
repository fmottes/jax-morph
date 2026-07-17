import io
import json
import math
import os
import struct
import subprocess
import sys

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jax_morph as jxm
from jax_morph.core import serialization as ser

_CHECKPOINT_SPEC = jxm.StateFieldSpec('checkpoint_value', dtype=bool)


def _manifest(kind='test'):
    return {
        'format': 'jax-morph-artifact',
        'format_version': 1,
        'kind': kind,
        'jax_morph_version': 'test',
    }


def test_persistence_public_api_is_exported():
    for name in (
        'save_model',
        'load_model',
        'save_state',
        'load_state',
        'save_trajectory',
        'load_trajectory',
        'TrajectoryRecord',
    ):
        assert name in jxm.__all__
        assert hasattr(jxm, name)


class _CheckpointStep(jxm.SimulationStep):
    step_type = jxm.StepType.DYNAMIC
    weight: object
    enabled: bool
    count: int
    offset: float
    phase: complex
    tag: str = eqx.field(static=True)
    spec: object = eqx.field(static=True)
    transform: object = eqx.field(static=True)

    def __init__(
        self,
        weight,
        *,
        enabled=True,
        count=2,
        offset=0.25,
        phase=1.0 + 2.0j,
        tag='checkpoint',
        spec=_CHECKPOINT_SPEC,
        transform=math.sin,
    ):
        self.weight = weight
        self.enabled = enabled
        self.count = count
        self.offset = offset
        self.phase = phase
        self.tag = tag
        self.spec = spec
        self.transform = transform

    def state_writes(self):
        return (jxm.RADIUS,)

    def __call__(self, state, *, dt, key):
        return state.deltas(radius=dt * self.weight * jnp.ones_like(state.radius))


def _checkpoint_model(weight, **kwargs):
    return jxm.Model([_CheckpointStep(weight, **kwargs)])


class _Pair(eqx.Module):
    left: jax.Array
    right: jax.Array


class _StateRequirements:
    def state_requires(self):
        return (
            jxm.StateFieldSpec('vector', shape=(2,), dtype=jnp.dtype('float32')),
            jxm.StateFieldSpec('grid', shape=(2, 3), scope='global', dtype=jnp.float32),
        )


def test_supported_payload_leaves_round_trip(tmp_path):
    leaves = (
        jnp.arange(3, dtype=jnp.float32),
        True,
        7,
        1.25,
        2.0 - 3.0j,
    )
    entries = [ser._leaf_entry(leaf) for leaf in leaves]
    path = tmp_path / 'payload.jxm'
    ser._write_container(path, _manifest(), leaves)

    with ser._read_container(path, expected_kind='test') as payload:
        restored = ser._read_payload(payload, entries)

    assert isinstance(restored[0], jax.Array)
    np.testing.assert_array_equal(restored[0], leaves[0])
    assert restored[1:] == leaves[1:]
    assert [type(value) for value in restored[1:]] == [bool, int, float, complex]


def test_save_transfers_each_jax_payload_leaf_to_host_once(tmp_path, monkeypatch):
    calls = 0
    original_device_get = ser.jax.device_get

    def counting_device_get(value):
        nonlocal calls
        calls += 1
        return original_device_get(value)

    monkeypatch.setattr(ser.jax, 'device_get', counting_device_get)
    ser._write_container(tmp_path / 'once.jxm', _manifest(), (jnp.arange(4),))
    assert calls == 1


def test_unsupported_payload_leaves_are_rejected(tmp_path):
    with pytest.raises(TypeError, match='object-dtype'):
        ser._leaf_entry(np.array([object()], dtype=object))
    with pytest.raises(TypeError, match='unsupported'):
        ser._leaf_entry(np.asarray([1.0]))

    with pytest.raises(TypeError, match='unsupported'):
        ser._write_container(tmp_path / 'bad.jxm', _manifest(), ('not numeric',))


@pytest.mark.parametrize(
    ('contents', 'match'),
    [
        (b'not-a-jxm', 'magic'),
        (ser.MAGIC + struct.pack('>Q', 2) + b'{]', 'JSON'),
        (
            ser.MAGIC
            + struct.pack('>Q', len(json.dumps(_manifest(kind='other')).encode()))
            + json.dumps(_manifest(kind='other')).encode(),
            'kind',
        ),
        (
            ser.MAGIC
            + struct.pack('>Q', len(json.dumps({**_manifest(), 'format_version': 2}).encode()))
            + json.dumps({**_manifest(), 'format_version': 2}).encode(),
            'version',
        ),
    ],
)
def test_container_header_errors_are_focused(tmp_path, contents, match):
    path = tmp_path / 'bad.jxm'
    path.write_bytes(contents)
    with (
        pytest.raises(ser.PersistenceError, match=match),
        ser._read_container(path, expected_kind='test'),
    ):
        pass


def test_payload_truncation_and_trailing_bytes_are_rejected(tmp_path):
    leaves = (jnp.arange(2, dtype=jnp.float32),)
    entries = [ser._leaf_entry(leaves[0])]
    path = tmp_path / 'payload.jxm'
    ser._write_container(path, _manifest(), leaves)

    truncated = tmp_path / 'truncated.jxm'
    truncated.write_bytes(path.read_bytes()[:-2])
    with (
        pytest.raises(ser.PersistenceError, match='payload leaf 0'),
        ser._read_container(truncated, expected_kind='test') as payload,
    ):
        ser._read_payload(payload, entries)

    path.write_bytes(path.read_bytes() + b'extra')
    with (
        pytest.raises(ser.PersistenceError, match='trailing'),
        ser._read_container(path, expected_kind='test') as payload,
    ):
        ser._read_payload(payload, entries)


def test_atomic_write_replaces_only_after_success(tmp_path):
    path = tmp_path / 'artifact.jxm'
    path.write_bytes(b'old artifact')

    with pytest.raises(TypeError):
        ser._write_container(path, _manifest(), (object(),))
    assert path.read_bytes() == b'old artifact'
    assert not list(tmp_path.glob('.artifact.jxm.*.tmp'))

    ser._write_container(path, _manifest(), (jnp.asarray(3.0),))
    with ser._read_container(path, expected_kind='test') as payload:
        restored = ser._read_payload(payload, [ser._leaf_entry(jnp.asarray(3.0))])
    assert float(restored[0]) == 3.0


def test_model_checkpoint_restores_numeric_leaves_into_template(tmp_path, key):
    model = _checkpoint_model(
        jnp.asarray(1.5), enabled=False, count=9, offset=1.25, phase=-2.0 + 0.5j
    )
    template = _checkpoint_model(jnp.asarray(99.0), enabled=True, count=1, offset=0.0, phase=0.0j)
    path = tmp_path / 'model.jxm'

    ser.save_model(path, model)
    restored = ser.load_model(path, template)
    original_step = model.steps[0]
    restored_step = restored.steps[0]
    assert isinstance(restored_step.weight, jax.Array)
    np.testing.assert_array_equal(restored_step.weight, original_step.weight)
    for name, expected_type in [
        ('enabled', bool),
        ('count', int),
        ('offset', float),
        ('phase', complex),
    ]:
        assert getattr(restored_step, name) == getattr(original_step, name)
        assert type(getattr(restored_step, name)) is expected_type
    assert restored_step.spec.dtype is bool

    state = jxm.build_state_from_model(model).init_empty(capacity=2, n_space_dim=2, n_types=1)
    state = state.set('alive', state.alive.at[0].set(True))
    np.testing.assert_allclose(
        model(state, dt=0.1, key=key).radius,
        restored(state, dt=0.1, key=key).radius,
    )


def test_model_checkpoint_validates_template_before_reading_payload(tmp_path):
    path = tmp_path / 'model.jxm'
    ser.save_model(path, _checkpoint_model(jnp.asarray(1.0)))

    with pytest.raises(ser.PersistenceError, match='model structure mismatch'):
        ser.load_model(path, _checkpoint_model(jnp.asarray(1.0), tag='different'))


def test_model_manifest_uses_one_ordered_payload_template(tmp_path):
    model = _checkpoint_model(jnp.asarray(1.0))
    path = tmp_path / 'model.jxm'
    ser.save_model(path, model)

    with ser._read_container(path, expected_kind='model') as payload:
        manifest = payload.manifest
        assert (
            ser._without_payload_checksums(manifest['payload'])
            == manifest['model_signature']['leaves']
        )
        assert all(len(entry['sha256']) == 64 for entry in manifest['payload'])
        leaves = ser._read_payload(payload, manifest['payload'])
    assert len(leaves) == len(manifest['model_signature']['leaves'])


def test_model_checkpoint_rejects_non_addressable_static_callable(tmp_path):
    model = _checkpoint_model(jnp.asarray(1.0), transform=lambda x: x)
    with pytest.raises(TypeError, match='transform'):
        ser.save_model(tmp_path / 'lambda.jxm', model)


def test_state_snapshot_is_self_describing_and_reuses_generated_class(tmp_path):
    State = jxm.build_state_from_model(_StateRequirements(), name='Persistence')
    state = State.init_empty(capacity=3, n_space_dim=2, n_types=2)
    state = state.update(
        alive=state.alive.at[0].set(True),
        radius=state.radius.at[0].set(jnp.asarray(0.5, dtype=state.radius.dtype)),
        vector=state.vector.at[0].set(jnp.asarray([1.0, 2.0], dtype=state.vector.dtype)),
        grid=jnp.arange(6, dtype=jnp.float32).reshape(2, 3),
        t=jnp.asarray(1.25),
    )
    path = tmp_path / 'state.jxm'

    ser.save_state(path, state)
    restored = ser.load_state(path)

    assert type(restored) is type(state)
    assert restored.specs['alive'].dtype is bool
    assert isinstance(restored.specs['vector'].dtype, np.dtype)
    assert restored.specs['grid'].dtype is jnp.float32
    for name in state.specs:
        np.testing.assert_array_equal(restored[name], state[name])


@pytest.mark.parametrize('box', [10.0, (10.0, 20.0)])
def test_state_snapshot_reconstructs_periodic_space(tmp_path, box):
    State = jxm.build_state_from_model(_StateRequirements(), name='PeriodicPersistence')
    state = State.init_empty(capacity=2, n_space_dim=2, n_types=1, space=jxm.geometry.periodic(box))
    path = tmp_path / 'periodic.jxm'
    ser.save_state(path, state)
    restored = ser.load_state(path)

    point_a = jnp.asarray([9.0, 19.0])
    point_b = jnp.asarray([1.0, 1.0])
    np.testing.assert_allclose(
        restored.displacement(point_a, point_b), state.displacement(point_a, point_b)
    )


def test_custom_state_space_requires_explicit_space_at_load(tmp_path):
    State = jxm.build_state_from_model(_StateRequirements(), name='CustomPersistence')
    custom_space = (lambda a, b: a - b, lambda a, d: a + d)
    state = State.init_empty(capacity=2, n_space_dim=2, n_types=1, space=custom_space)
    path = tmp_path / 'custom.jxm'
    ser.save_state(path, state)

    with pytest.raises(ser.PersistenceError, match='space='):
        ser.load_state(path)
    restored = ser.load_state(path, space=custom_space)
    np.testing.assert_array_equal(restored.position, state.position)


def test_state_snapshot_rejects_a_stacked_history(tmp_path):
    model = _checkpoint_model(jnp.asarray(1.0))
    state = jxm.build_state_from_model(model).init_empty(capacity=2, n_space_dim=2, n_types=1)
    history = jxm.simulate(model, state, n_steps=1, dt=0.1, history=True)
    with pytest.raises(ValueError, match='save_trajectory'):
        ser.save_state(tmp_path / 'history.jxm', history)


def test_trajectory_round_trips_complete_history_dt_and_provenance(tmp_path):
    model = _checkpoint_model(jnp.asarray(1.5))
    state = jxm.build_state_from_model(model).init_empty(capacity=2, n_space_dim=2, n_types=1)
    state = state.update(alive=state.alive.at[0].set(True), t=jnp.asarray(2.0))
    history = jxm.simulate(model, state, n_steps=2, dt=0.25, history=True)
    path = tmp_path / 'trajectory.jxm'

    ser.save_trajectory(path, history, 0.25, provenance={'run': 'test', 'seed': 0})
    record = ser.load_trajectory(path)

    assert isinstance(record.dt, jax.Array)
    assert record.dt.dtype == history.t.dtype
    assert record.provenance == {'run': 'test', 'seed': 0}
    for name in history.specs:
        np.testing.assert_array_equal(record.history[name], history[name])
    np.testing.assert_allclose(
        jxm.trajectory_logp(model, record.history, record.dt),
        jxm.trajectory_logp(model, history, jnp.asarray(0.25, dtype=history.t.dtype)),
    )


def test_trajectory_payload_places_dt_last_and_rejects_unstacked_input(tmp_path):
    model = _checkpoint_model(jnp.asarray(1.0))
    state = jxm.build_state_from_model(model).init_empty(capacity=2, n_space_dim=2, n_types=1)
    history = jxm.simulate(model, state, n_steps=1, dt=0.1, history=True)
    path = tmp_path / 'trajectory.jxm'
    ser.save_trajectory(path, history, jnp.asarray(0.1, dtype=jnp.float32))

    with ser._read_container(path, expected_kind='trajectory') as payload:
        entries = payload.manifest['payload']
        assert [entry['name'] for entry in entries[:-1]] == payload.manifest['state']['field_names']
        assert entries[-1]['name'] == 'dt'
        restored = ser._read_payload(payload, entries)
    assert restored[-1].dtype == jnp.float32

    with pytest.raises(ValueError, match='save_state'):
        ser.save_trajectory(tmp_path / 'not-history.jxm', state, 0.1)


# --- shared helpers for the scoring, gradient, and tampering coverage below --------------------

_U = jxm.StateFieldSpec('u', shape=(), heritable=False)
_DEFAULT_LOG_STD = float(np.log(0.4))


def _gaussian_logp(x, mean, std):
    return -0.5 * jnp.log(2 * jnp.pi) - jnp.log(std) - 0.5 * ((x - mean) / std) ** 2


class _Kick(jxm.StochasticStep):
    """Reparameterized stochastic kick used to score loaded trajectories under any candidate."""

    step_type = jxm.StepType.DYNAMIC
    log_std: jax.Array
    prefix: str = eqx.field(static=True, default='kick')

    def state_writes(self):
        return (_U,)

    def trace_writes(self):
        return (
            jxm.StateFieldSpec(f'{self.prefix}_xi', shape=(), heritable=False),
            jxm.StateFieldSpec(f'{self.prefix}_dx', shape=(), heritable=False),
        )

    def _dist(self, state, dt):
        return jnp.zeros_like(state.u), jnp.exp(self.log_std) * jnp.ones_like(state.u)

    def sample_trace(self, state, *, dt, key):
        return {f'{self.prefix}_xi': jax.random.normal(key, state.u.shape)}

    def replay(self, state, trace, *, dt, pathwise):
        mean, std = self._dist(state, dt)
        xi = trace[f'{self.prefix}_xi']
        dx = mean + std * xi if pathwise else trace[f'{self.prefix}_dx']
        return state.deltas(**{'u': dx, f'{self.prefix}_xi': xi, f'{self.prefix}_dx': dx})

    def logp(self, state, trace, dt):
        mean, std = self._dist(state, dt)
        alive = state.alive.astype(std.dtype)
        return jnp.sum(_gaussian_logp(trace[f'{self.prefix}_dx'], mean, std) * alive)


def _kick_model(log_std=_DEFAULT_LOG_STD, prefix='kick'):
    return jxm.Model([_Kick(log_std=jnp.asarray(log_std), prefix=prefix)])


def _kick_setup(log_std=_DEFAULT_LOG_STD):
    model = _kick_model(log_std)
    state = jxm.build_state_from_model(model).init_empty(capacity=3, n_space_dim=2, n_types=1)
    state = state.update(alive=state.alive.at[jnp.array([0, 1])].set(True), t=jnp.asarray(1.0))
    return model, state


def _rewrite_manifest(path, mutate):
    """Decode a container's JSON manifest, mutate it in place, and rewrite the file."""
    raw = path.read_bytes()
    header = len(ser.MAGIC)
    length = struct.unpack('>Q', raw[header : header + 8])[0]
    start = header + 8
    manifest = json.loads(raw[start : start + length])
    mutate(manifest)
    encoded = json.dumps(manifest).encode('utf-8')
    path.write_bytes(ser.MAGIC + struct.pack('>Q', len(encoded)) + encoded + raw[start + length :])


# --- model checkpoints (Task 9.2 gaps) --------------------------------------------------------


def test_model_checkpoint_array_parameters_stay_differentiable(tmp_path):
    path = tmp_path / 'model.jxm'
    ser.save_model(path, _checkpoint_model(jnp.asarray(1.5)))
    restored = ser.load_model(path, _checkpoint_model(jnp.asarray(0.0)))

    assert isinstance(restored.steps[0].weight, jax.Array)
    grad = eqx.filter_grad(lambda m: m.steps[0].weight ** 2)(restored)
    assert float(grad.steps[0].weight) == pytest.approx(2 * 1.5)


def test_model_signature_detects_payload_leaf_reordering(tmp_path):
    path = tmp_path / 'model.jxm'
    model = _Pair(jnp.asarray(1.0), jnp.asarray(2.0))
    ser.save_model(path, model)

    raw = path.read_bytes()
    header = len(ser.MAGIC)
    manifest_length = struct.unpack('>Q', raw[header : header + 8])[0]
    payload_start = header + 8 + manifest_length
    stream = io.BytesIO(raw[payload_start:])
    records = []
    for _ in range(2):
        start = stream.tell()
        np.load(stream, allow_pickle=False)
        records.append(raw[payload_start + start : payload_start + stream.tell()])
    path.write_bytes(raw[:payload_start] + records[1] + records[0])

    with pytest.raises(ser.PersistenceError, match='integrity'):
        ser.load_model(path, _Pair(jnp.asarray(0.0), jnp.asarray(0.0)))


def test_model_signature_pins_equinox_auxiliary_layout():
    signature, _, _ = ser._model_signature(_checkpoint_model(jnp.asarray(1.0)))
    model_node = signature['tree']
    step_node = model_node['children'][0]['children'][0]

    assert model_node['auxiliary']['items'] == [{'kind': 'equinox_missing'}]
    assert [item['kind'] for item in step_node['auxiliary']['items']] == [
        'equinox_missing',
        'string',
        'state_field_spec',
        'callable',
    ]


def test_model_signature_distinguishes_declared_dtype_representation(tmp_path):
    path = tmp_path / 'model.jxm'
    ser.save_model(path, _checkpoint_model(jnp.asarray(1.0)))  # spec dtype is the literal bool

    dtype_object_spec = jxm.StateFieldSpec('checkpoint_value', dtype=jnp.dtype('bool'))
    with pytest.raises(ser.PersistenceError, match='model structure mismatch'):
        ser.load_model(path, _checkpoint_model(jnp.asarray(1.0), spec=dtype_object_spec))
    # The identical representation still loads.
    ser.load_model(path, _checkpoint_model(jnp.asarray(1.0)))


def test_load_model_compares_signatures_without_importing_artifact_modules(tmp_path):
    path = tmp_path / 'model.jxm'
    ser.save_model(path, _checkpoint_model(jnp.asarray(1.0)))
    bogus_module = 'jax_morph_absent_artifact_module'
    assert bogus_module not in sys.modules

    _rewrite_manifest(
        path, lambda m: m['model_signature']['tree'].__setitem__('type', f'{bogus_module}:Fake')
    )
    with pytest.raises(ser.PersistenceError, match='model structure mismatch'):
        ser.load_model(path, _checkpoint_model(jnp.asarray(1.0)))
    assert bogus_module not in sys.modules


# --- state snapshots (Task 9.3 gaps) ----------------------------------------------------------


def test_loaded_state_continues_simulation_matching_original(tmp_path, key):
    model = _checkpoint_model(jnp.asarray(0.7))
    state = jxm.build_state_from_model(model).init_empty(capacity=3, n_space_dim=2, n_types=1)
    state = state.update(
        alive=state.alive.at[0].set(True),
        radius=state.radius.at[0].set(jnp.asarray(0.4)),
    )
    path = tmp_path / 'state.jxm'
    ser.save_state(path, state)
    restored = ser.load_state(path)

    original = jxm.simulate(model, state, n_steps=3, dt=0.2, key=key)
    continued = jxm.simulate(model, restored, n_steps=3, dt=0.2, key=key)
    np.testing.assert_allclose(continued.radius, original.radius)
    np.testing.assert_allclose(continued.position, original.position)


def test_supplied_builtin_space_mismatch_is_rejected(tmp_path):
    State = jxm.build_state_from_model(_StateRequirements(), name='SpaceMismatch')
    state = State.init_empty(
        capacity=2, n_space_dim=2, n_types=1, space=jxm.geometry.periodic(10.0)
    )
    path = tmp_path / 'space.jxm'
    ser.save_state(path, state)

    with pytest.raises(ser.PersistenceError, match='does not match the saved built-in'):
        ser.load_state(path, space=jxm.geometry.periodic(20.0))
    assert ser.load_state(path, space=jxm.geometry.periodic(10.0)).n_cells == 2


def test_periodic_like_space_outside_cache_saves_as_custom(tmp_path):
    box = jnp.asarray([5.0, 5.0])
    displacement = lambda a, b: (a - b) - box * jnp.round((a - b) / box)  # noqa: E731
    shift = lambda r, d: jnp.mod(r + d, box)  # noqa: E731
    State = jxm.build_state_from_model(_StateRequirements(), name='UncachedSpace')
    state = State.init_empty(capacity=2, n_space_dim=2, n_types=1, space=(displacement, shift))
    path = tmp_path / 'uncached.jxm'
    ser.save_state(path, state)

    with ser._read_container(path, expected_kind='state') as payload:
        assert payload.manifest['state']['space']['kind'] == 'custom'
    with pytest.raises(ser.PersistenceError, match='space='):
        ser.load_state(path)


def test_periodic_box_dtype_is_validated_on_load(tmp_path):
    State = jxm.build_state_from_model(_StateRequirements(), name='PeriodicDtype')
    state = State.init_empty(
        capacity=2, n_space_dim=2, n_types=1, space=jxm.geometry.periodic(10.0)
    )
    path = tmp_path / 'periodic-dtype.jxm'
    ser.save_state(path, state)

    def change_box_dtype(manifest):
        manifest['state']['space']['box_dtype'] = jnp.dtype('float32').str

    _rewrite_manifest(path, change_box_dtype)
    with pytest.raises(ser.PersistenceError, match='box dtype'):
        ser.load_state(path)


def test_state_artifact_rejects_malformed_schema(tmp_path):
    State = jxm.build_state_from_model(_StateRequirements(), name='Malformed')
    state = State.init_empty(capacity=2, n_space_dim=2, n_types=1)

    def duplicate(manifest):
        names = manifest['state']['field_names']
        names[1] = names[0]

    def bad_scope(manifest):
        manifest['state']['specs'][0]['scope'] = 'nowhere'

    def conflicting_specs(manifest):
        manifest['state']['specs'][1]['name'] = manifest['state']['specs'][0]['name']

    duplicate_path = tmp_path / 'duplicate.jxm'
    ser.save_state(duplicate_path, state)
    _rewrite_manifest(duplicate_path, duplicate)
    with pytest.raises(ser.PersistenceError, match='duplicate'):
        ser.load_state(duplicate_path)

    scope_path = tmp_path / 'scope.jxm'
    ser.save_state(scope_path, state)
    _rewrite_manifest(scope_path, bad_scope)
    with pytest.raises(ser.PersistenceError, match='heritable/scope'):
        ser.load_state(scope_path)

    conflict_path = tmp_path / 'conflict.jxm'
    ser.save_state(conflict_path, state)
    _rewrite_manifest(conflict_path, conflicting_specs)
    with pytest.raises(ser.PersistenceError, match='field specs'):
        ser.load_state(conflict_path)


def test_save_state_rejects_a_missing_declared_field(tmp_path):
    State = jxm.build_state_from_model(_StateRequirements(), name='MissingField')
    state = State.init_empty(capacity=2, n_space_dim=2, n_types=1)
    object.__delattr__(state, 'grid')

    with pytest.raises(TypeError, match="missing declared field 'grid'"):
        ser.save_state(tmp_path / 'missing.jxm', state)


def test_float64_downcast_is_rejected_under_x64_disabled(tmp_path):
    state = jxm.build_state_from_model(jxm.Model([])).init_empty(
        capacity=2, n_space_dim=2, n_types=1
    )
    assert state.position.dtype == jnp.float64
    path = tmp_path / 'f64.jxm'
    ser.save_state(path, state)

    script = (
        'import sys\n'
        'import jax.numpy as jnp\n'
        'assert jnp.asarray(1.0).dtype == jnp.float32, "x64 unexpectedly enabled"\n'
        'from jax_morph.core import serialization as ser\n'
        'try:\n'
        '    ser.load_state(sys.argv[1])\n'
        'except ser.PersistenceError as error:\n'
        '    assert "loaded exactly" in str(error), str(error)\n'
        '    print("REJECTED")\n'
        'else:\n'
        '    raise SystemExit("float64 artifact loaded under x64-disabled")\n'
    )
    result = subprocess.run(
        [sys.executable, '-c', script, str(path)],
        capture_output=True,
        text=True,
        env={**os.environ, 'JAX_ENABLE_X64': '0'},
    )
    assert result.returncode == 0, result.stderr
    assert 'REJECTED' in result.stdout


# --- complete trajectories (Task 9.4 gaps) ----------------------------------------------------


def test_zero_step_trajectory_is_valid_and_scores_empty(tmp_path, key):
    model, state = _kick_setup()
    history = jxm.simulate(model, state, n_steps=0, dt=0.25, key=key, history=True)
    path = tmp_path / 'zero.jxm'
    ser.save_trajectory(path, history, 0.25)

    record = ser.load_trajectory(path)
    assert record.history.t.shape == (1,)
    assert jxm.trajectory_logp(model, record.history, record.dt).shape == (0,)


def test_trajectory_scores_and_gradients_survive_round_trip(tmp_path, key):
    model, state = _kick_setup()
    history = jxm.simulate(model, state, n_steps=4, dt=0.25, key=key, history=True)
    path = tmp_path / 'traj.jxm'
    ser.save_trajectory(path, history, 0.25)
    record = ser.load_trajectory(path)

    dt = jnp.asarray(0.25, dtype=history.t.dtype)
    np.testing.assert_allclose(
        jxm.trajectory_logp(model, record.history, record.dt),
        jxm.trajectory_logp(model, history, dt),
    )

    def log_std_grad(hist, step):
        return (
            eqx.filter_grad(lambda m: jxm.trajectory_logp(m, hist, step).sum())(model)
            .steps[0]
            .log_std
        )

    grad_before = log_std_grad(history, dt)
    grad_after = log_std_grad(record.history, record.dt)
    assert float(jnp.abs(grad_before)) > 0
    np.testing.assert_allclose(grad_after, grad_before)


def test_trajectory_is_scoreable_under_a_different_candidate_model(tmp_path, key):
    reference, state = _kick_setup(log_std=np.log(0.4))
    history = jxm.simulate(reference, state, n_steps=3, dt=0.2, key=key, history=True)
    path = tmp_path / 'traj.jxm'
    ser.save_trajectory(path, history, 0.2, provenance={'generating_model': 'reference'})
    record = ser.load_trajectory(path)

    candidate = _kick_model(log_std=np.log(1.3))  # different parameters, mismatched provenance
    candidate_terms = jxm.trajectory_logp(candidate, record.history, record.dt)
    reference_terms = jxm.trajectory_logp(reference, record.history, record.dt)
    assert candidate_terms.shape == reference_terms.shape
    assert not bool(jnp.allclose(candidate_terms, reference_terms))


def test_candidate_requiring_missing_trace_field_fails_diagnostically(tmp_path, key):
    reference, state = _kick_setup()
    history = jxm.simulate(reference, state, n_steps=2, dt=0.25, key=key, history=True)
    path = tmp_path / 'traj.jxm'
    ser.save_trajectory(path, history, 0.25)
    record = ser.load_trajectory(path)

    candidate = _kick_model(prefix='other')  # requires 'other_xi'/'other_dx', absent in the history
    with pytest.raises(AttributeError, match='other'):
        jxm.trajectory_logp(candidate, record.history, record.dt)


def test_trajectory_dt_is_normalised_and_invalid_dt_is_rejected(tmp_path, key):
    model, state = _kick_setup()
    history = jxm.simulate(model, state, n_steps=2, dt=0.25, key=key, history=True)

    typed = tmp_path / 'typed.jxm'
    ser.save_trajectory(typed, history, jnp.asarray(0.25, dtype=jnp.float32))
    assert ser.load_trajectory(typed).dt.dtype == jnp.float32

    python = tmp_path / 'python.jxm'
    ser.save_trajectory(python, history, 0.25)
    record = ser.load_trajectory(python)
    assert isinstance(record.dt, jax.Array) and record.dt.dtype == history.t.dtype

    for bad in (
        True,
        0.25 + 1j,
        jnp.asarray(True),
        jnp.asarray(0.25 + 1j),
        jnp.asarray([0.25, 0.25]),
        jnp.asarray(jnp.inf),
    ):
        with pytest.raises(ValueError):
            ser.save_trajectory(tmp_path / 'bad.jxm', history, bad)
    with pytest.raises(ValueError, match='increments'):
        ser.save_trajectory(tmp_path / 'inconsistent.jxm', history, 0.5)
    with pytest.raises(TypeError, match='JSON'):
        ser.save_trajectory(tmp_path / 'provenance.jxm', history, 0.25, provenance={'x': object()})


def test_strided_history_is_rejected(tmp_path, key):
    model, state = _kick_setup()
    history = jxm.simulate(model, state, n_steps=4, dt=0.25, key=key, history=True)
    strided = jax.tree_util.tree_map(lambda leaf: leaf[::2], history)  # drops intermediate frames
    with pytest.raises(ValueError, match='increments'):
        ser.save_trajectory(tmp_path / 'strided.jxm', strided, 0.25)
