"""Versioned, non-executable persistence helpers for jax-morph artifacts.

The public model, state, and trajectory entry points are built on the small container and numeric
leaf codec in this module.  The format uses JSON metadata and NPY payload leaves; it never uses
pickle or executes artifact-supplied code.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import struct
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import BinaryIO

import jax
import jax.numpy as jnp
import numpy as np

from .geometry import _space_descriptor, _space_from_descriptor
from .state import BaseState, StateFieldSpec, _make_state_cls

__all__ = [
    'PersistenceError',
    'TrajectoryRecord',
    'load_model',
    'load_state',
    'load_trajectory',
    'save_model',
    'save_state',
    'save_trajectory',
]


MAGIC = b'JAXMORPH\x00'
FORMAT_NAME = 'jax-morph-artifact'
FORMAT_VERSION = 1
_MAX_MANIFEST_BYTES = 16 * 1024 * 1024
_MAX_LEAF_BYTES = 1 << 40
_NUMERIC_CATEGORIES = {
    'jax_array',
    'python_bool',
    'python_int',
    'python_float',
    'python_complex',
}
_JAX_SCALAR_TYPES = {
    jnp.dtype(scalar_type).str: scalar_type
    for scalar_type in (
        jnp.bool_,
        jnp.int8,
        jnp.int16,
        jnp.int32,
        jnp.int64,
        jnp.uint8,
        jnp.uint16,
        jnp.uint32,
        jnp.uint64,
        jnp.float16,
        jnp.float32,
        jnp.float64,
        jnp.complex64,
        jnp.complex128,
    )
}


class PersistenceError(ValueError):
    """Raised when a persistence artifact is malformed or incompatible."""


@dataclass(frozen=True)
class _ContainerPayload:
    """Validated manifest plus a file object positioned at its numeric payload."""

    manifest: dict[str, object]
    file: BinaryIO

    def __getattr__(self, name: str) -> object:
        """Delegate file methods needed by the NPY codec."""
        return getattr(self.file, name)


def _canonical_json(value: object) -> bytes:
    """Encode JSON deterministically and reject non-finite or unsupported values."""
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(',', ':'),
            sort_keys=True,
        ).encode('utf-8')
    except (TypeError, ValueError) as error:
        raise TypeError(f'Persistence manifest must be finite JSON-safe data: {error}') from error


def _leaf_entry(leaf: object) -> dict[str, object]:
    """Return the manifest descriptor for one supported numeric payload leaf."""
    category: str
    if isinstance(leaf, jax.Array):
        dtype = np.dtype(leaf.dtype)
        if dtype.hasobject:
            raise TypeError('object-dtype arrays are not supported in persistence payloads')
        return {
            'category': 'jax_array',
            'shape': list(leaf.shape),
            'dtype': dtype.str,
        }
    elif type(leaf) is bool:
        category = 'python_bool'
        array = np.asarray(leaf)
    elif type(leaf) is int:
        category = 'python_int'
        array = np.asarray(leaf)
    elif type(leaf) is float:
        category = 'python_float'
        array = np.asarray(leaf)
    elif type(leaf) is complex:
        category = 'python_complex'
        array = np.asarray(leaf)
    elif isinstance(leaf, np.ndarray) and leaf.dtype.hasobject:
        raise TypeError('object-dtype arrays are not supported in persistence payloads')
    else:
        raise TypeError(
            f'unsupported persistence payload leaf {type(leaf).__module__}.'
            f'{type(leaf).__qualname__}; ordinary fields must be a jax.Array or Python numeric '
            'scalar, and structural data must use eqx.field(static=True)'
        )

    if array.dtype.hasobject:
        raise TypeError('object-dtype arrays are not supported in persistence payloads')
    return {
        'category': category,
        'shape': list(array.shape),
        'dtype': array.dtype.str,
    }


def _validate_leaf_entry(entry: object, *, index: int) -> tuple[str, tuple[int, ...], np.dtype]:
    """Validate one payload descriptor without allocating its declared shape."""
    if not isinstance(entry, dict):
        raise PersistenceError(f'payload leaf {index} descriptor must be an object')
    category = entry.get('category')
    if category not in _NUMERIC_CATEGORIES:
        raise PersistenceError(f'payload leaf {index} has unsupported category {category!r}')
    shape = entry.get('shape')
    if not isinstance(shape, list) or any(type(dim) is not int or dim < 0 for dim in shape):
        raise PersistenceError(f'payload leaf {index} has an invalid shape descriptor')
    if category != 'jax_array' and shape:
        raise PersistenceError(f'payload leaf {index} Python scalar must have shape []')
    try:
        dtype = np.dtype(entry.get('dtype'))
    except TypeError as error:
        raise PersistenceError(f'payload leaf {index} has an invalid dtype descriptor') from error
    if dtype.hasobject:
        raise PersistenceError(f'payload leaf {index} uses unsupported object dtype')
    try:
        bytes_required = int(np.prod(shape, dtype=object)) * dtype.itemsize
    except (OverflowError, TypeError) as error:
        raise PersistenceError(f'payload leaf {index} has an impossible shape/dtype') from error
    if bytes_required > _MAX_LEAF_BYTES:
        raise PersistenceError(f'payload leaf {index} declares more than {_MAX_LEAF_BYTES} bytes')
    return category, tuple(shape), dtype


def _check_jax_dtype(dtype: np.dtype, *, index: int) -> None:
    """Reject a dtype JAX would silently narrow under the current configuration."""
    try:
        loaded_dtype = np.dtype(jnp.asarray(np.empty((), dtype=dtype)).dtype)
    except TypeError as error:
        raise PersistenceError(
            f'payload leaf {index} dtype {dtype.str!r} is unsupported by JAX'
        ) from error
    if loaded_dtype != dtype:
        raise PersistenceError(
            f'payload leaf {index} dtype {dtype.str!r} cannot be loaded exactly; '
            'enable the required JAX dtype support (for example JAX x64)'
        )


def _write_payload(file: BinaryIO, arrays: Sequence[np.ndarray]) -> None:
    """Write prepared host arrays as a sequence of pickle-free NPY records."""
    for index, array in enumerate(arrays):
        try:
            np.save(file, array, allow_pickle=False)
        except (OSError, ValueError, TypeError) as error:
            raise PersistenceError(f'failed to write payload leaf {index}: {error}') from error


def _read_payload(file: BinaryIO, entries: Sequence[object]) -> tuple[object, ...]:
    """Read and restore manifest-described leaves, requiring exact payload EOF."""
    restored: list[object] = []
    for index, entry in enumerate(entries):
        category, shape, dtype = _validate_leaf_entry(entry, index=index)
        if category == 'jax_array':
            _check_jax_dtype(dtype, index=index)
        try:
            array = np.load(file, allow_pickle=False)
        except (OSError, ValueError, EOFError) as error:
            raise PersistenceError(f'failed to read payload leaf {index}: {error}') from error
        if not isinstance(array, np.ndarray):
            raise PersistenceError(f'payload leaf {index} is not an NPY array')
        if tuple(array.shape) != shape:
            raise PersistenceError(
                f'payload leaf {index} shape mismatch: expected {shape}, got {tuple(array.shape)}'
            )
        if array.dtype != dtype:
            raise PersistenceError(
                f'payload leaf {index} dtype mismatch: expected {dtype.str!r}, got {array.dtype.str!r}'
            )
        checksum = entry.get('sha256')
        if checksum is not None:
            if not isinstance(checksum, str) or len(checksum) != 64:
                raise PersistenceError(f'payload leaf {index} has an invalid SHA-256 descriptor')
            actual_checksum = hashlib.sha256(array.tobytes(order='C')).hexdigest()
            if actual_checksum != checksum:
                raise PersistenceError(f'payload leaf {index} failed its SHA-256 integrity check')
        if category == 'jax_array':
            restored.append(jnp.asarray(array))
        elif category == 'python_bool':
            restored.append(bool(array.item()))
        elif category == 'python_int':
            restored.append(int(array.item()))
        elif category == 'python_float':
            restored.append(float(array.item()))
        else:
            restored.append(complex(array.item()))
    if file.read(1):
        raise PersistenceError('artifact payload has trailing bytes or extra leaves')
    return tuple(restored)


def _validate_common_manifest(manifest: object, *, expected_kind: str) -> dict[str, object]:
    """Validate the shared versioned manifest header and selected artifact kind."""
    if not isinstance(manifest, dict):
        raise PersistenceError('artifact manifest must be a JSON object')
    if manifest.get('format') != FORMAT_NAME:
        raise PersistenceError(f'unsupported artifact format {manifest.get("format")!r}')
    if manifest.get('format_version') != FORMAT_VERSION:
        raise PersistenceError(
            f'unsupported artifact format version {manifest.get("format_version")!r}; '
            f'only version {FORMAT_VERSION} is supported'
        )
    if manifest.get('kind') != expected_kind:
        raise PersistenceError(
            f'artifact kind {manifest.get("kind")!r} does not match expected kind {expected_kind!r}'
        )
    if not isinstance(manifest.get('jax_morph_version'), str):
        raise PersistenceError('artifact manifest is missing string jax_morph_version')
    return manifest


@contextmanager
def _read_container(
    path: str | os.PathLike[str], *, expected_kind: str
) -> Iterator[_ContainerPayload]:
    """Open and validate a container header, yielding its positioned payload stream."""
    source = Path(path)
    try:
        file = source.open('rb')
    except OSError as error:
        raise PersistenceError(f'could not open artifact {source}: {error}') from error
    try:
        if file.read(len(MAGIC)) != MAGIC:
            raise PersistenceError(f'artifact {source} has invalid magic bytes')
        raw_length = file.read(8)
        if len(raw_length) != 8:
            raise PersistenceError(f'artifact {source} is truncated before manifest length')
        manifest_length = struct.unpack('>Q', raw_length)[0]
        if manifest_length > _MAX_MANIFEST_BYTES:
            raise PersistenceError(
                f'artifact {source} manifest exceeds {_MAX_MANIFEST_BYTES} byte safety limit'
            )
        raw_manifest = file.read(manifest_length)
        if len(raw_manifest) != manifest_length:
            raise PersistenceError(f'artifact {source} is truncated in its manifest')
        try:
            manifest = json.loads(raw_manifest.decode('utf-8'))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise PersistenceError(
                f'artifact {source} has invalid JSON manifest: {error}'
            ) from error
        _validate_common_manifest(manifest, expected_kind=expected_kind)
        yield _ContainerPayload(manifest, file)
    finally:
        file.close()


def _write_container(
    path: str | os.PathLike[str], manifest: dict[str, object], leaves: Sequence[object]
) -> None:
    """Atomically write a validated header and ordered NPY payload leaves to ``path``."""
    _validate_common_manifest(manifest, expected_kind=str(manifest.get('kind')))
    prepared = []
    descriptors = []
    for leaf in leaves:
        descriptors.append(_leaf_entry(leaf))
        if isinstance(leaf, jax.Array):
            leaf.block_until_ready()
            prepared.append(np.asarray(jax.device_get(leaf)))
        else:
            prepared.append(np.asarray(leaf))
    payload_entries = manifest.get('payload')
    if payload_entries is not None:
        if not isinstance(payload_entries, list) or len(payload_entries) != len(prepared):
            raise PersistenceError('manifest payload entry count does not match numeric leaves')
        checked_entries = []
        for index, (entry, descriptor, array) in enumerate(
            zip(payload_entries, descriptors, prepared, strict=True)
        ):
            if not isinstance(entry, dict) or any(
                entry.get(key) != descriptor[key] for key in ('category', 'shape', 'dtype')
            ):
                raise PersistenceError(f'manifest payload entry {index} does not match its leaf')
            checked_entries.append(
                {**entry, 'sha256': hashlib.sha256(array.tobytes(order='C')).hexdigest()}
            )
        manifest = {**manifest, 'payload': checked_entries}
    encoded_manifest = _canonical_json(manifest)
    if len(encoded_manifest) > _MAX_MANIFEST_BYTES:
        raise PersistenceError(f'manifest exceeds {_MAX_MANIFEST_BYTES} byte safety limit')

    destination = Path(path)
    temporary_name: str | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=destination.parent,
            prefix=f'.{destination.name}.',
            suffix='.tmp',
        )
        with os.fdopen(descriptor, 'wb') as file:
            file.write(MAGIC)
            file.write(struct.pack('>Q', len(encoded_manifest)))
            file.write(encoded_manifest)
            _write_payload(file, prepared)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_name, destination)
        temporary_name = None
    except OSError as error:
        raise PersistenceError(f'could not write artifact {destination}: {error}') from error
    finally:
        if temporary_name is not None:
            with suppress(FileNotFoundError):
                os.unlink(temporary_name)


def _package_version() -> str:
    """Return the installed package version without importing the package root."""
    try:
        return importlib.metadata.version('jax-morph')
    except importlib.metadata.PackageNotFoundError:
        return 'unknown'


def _qualified_name(value: object, *, path: str, require_importable: bool = False) -> str:
    """Return a stable module-qualified identity, rejecting unstable callable names."""
    module = getattr(value, '__module__', None)
    qualname = getattr(value, '__qualname__', None)
    if not isinstance(module, str) or not isinstance(qualname, str):
        raise TypeError(f'{path} has no stable module-qualified identity')
    if require_importable and ('<locals>' in qualname or '<lambda>' in qualname):
        raise TypeError(f'{path} is not an import-addressable module-level callable')
    return f'{module}:{qualname}'


def _encode_spec_dtype(dtype: object, *, path: str) -> dict[str, object]:
    """Encode a StateFieldSpec dtype without collapsing its Python representation category."""
    if dtype is None:
        return {'kind': 'none'}
    for builtin in (bool, int, float, complex):
        if dtype is builtin:
            return {'kind': 'python_builtin', 'name': builtin.__name__}
    for dtype_string, scalar_type in _JAX_SCALAR_TYPES.items():
        if dtype is scalar_type:
            return {'kind': 'jax_scalar_type', 'value': dtype_string}
    if isinstance(dtype, np.dtype):
        return {'kind': 'dtype_object', 'value': dtype.str}
    if isinstance(dtype, type):
        try:
            numpy_dtype = np.dtype(dtype)
        except TypeError:
            numpy_dtype = None
        if numpy_dtype is not None and numpy_dtype.type is dtype:
            return {'kind': 'scalar_type', 'value': numpy_dtype.str}
    if isinstance(dtype, str):
        return {'kind': 'string', 'value': dtype}
    raise TypeError(f'{path} has unsupported declared dtype {dtype!r}')


def _encode_state_field_spec(spec: StateFieldSpec, *, path: str) -> dict[str, object]:
    """Encode a StateFieldSpec with its declared dtype handled by the tagged codec."""
    return {
        'kind': 'state_field_spec',
        'name': spec.name,
        'shape': list(spec.shape),
        'dtype': _encode_spec_dtype(spec.dtype, path=f'{path}.dtype'),
        'heritable': spec.heritable,
        'default': _encode_static(spec.default, path=f'{path}.default'),
        'scope': spec.scope,
    }


def _encode_static(value: object, *, path: str) -> dict[str, object]:
    """Encode supported static PyTree auxiliary data without repr-based fallbacks."""
    value_type = type(value)
    if (
        value_type.__module__ == 'equinox._module._flatten'
        and value_type.__qualname__ == '_Missing'
    ):
        return {'kind': 'equinox_missing'}
    if value is None:
        return {'kind': 'none'}
    if type(value) is bool:
        return {'kind': 'bool', 'value': value}
    if type(value) is int:
        return {'kind': 'int', 'value': value}
    if type(value) is float:
        if not math.isfinite(value):
            raise TypeError(f'{path} contains a non-finite static float')
        return {'kind': 'float', 'value': value}
    if type(value) is complex:
        if not math.isfinite(value.real) or not math.isfinite(value.imag):
            raise TypeError(f'{path} contains a non-finite static complex value')
        return {'kind': 'complex', 'real': value.real, 'imag': value.imag}
    if type(value) is str:
        return {'kind': 'string', 'value': value}
    if isinstance(value, StateFieldSpec):
        return _encode_state_field_spec(value, path=path)
    if isinstance(value, np.dtype):
        return {'kind': 'dtype_object', 'value': value.str}
    if isinstance(value, Enum):
        return {
            'kind': 'enum',
            'type': _qualified_name(type(value), path=path),
            'name': value.name,
        }
    if isinstance(value, tuple):
        return {
            'kind': 'tuple',
            'items': [
                _encode_static(item, path=f'{path}[{index}]') for index, item in enumerate(value)
            ],
        }
    if isinstance(value, list):
        return {
            'kind': 'list',
            'items': [
                _encode_static(item, path=f'{path}[{index}]') for index, item in enumerate(value)
            ],
        }
    if isinstance(value, Mapping):
        items = [
            (
                _encode_static(key, path=f'{path}[key]'),
                _encode_static(item, path=f'{path}[{key!r}]'),
            )
            for key, item in value.items()
        ]
        items.sort(key=lambda pair: _canonical_json(pair[0]))
        return {'kind': 'dict', 'items': [[key, item] for key, item in items]}
    if isinstance(value, type):
        return {'kind': 'class', 'value': _qualified_name(value, path=path)}
    if inspect.isroutine(value):
        return {
            'kind': 'callable',
            'value': _qualified_name(value, path=path, require_importable=True),
        }
    if callable(value):
        raise TypeError(f'{path} has unsupported opaque callable {type(value).__qualname__}')
    if is_dataclass(value) and not isinstance(value, type):
        return {
            'kind': 'dataclass',
            'type': _qualified_name(type(value), path=path),
            'fields': [
                [
                    field.name,
                    _encode_static(getattr(value, field.name), path=f'{path}.{field.name}'),
                ]
                for field in fields(value)
            ],
        }
    raise TypeError(
        f'{path} has unsupported static value {type(value).__module__}.{type(value).__qualname__}'
    )


def _encode_tree_auxiliary(auxiliary: object, node_type: object) -> dict[str, object]:
    """Encode a node's PyTree auxiliary data with dataclass field names in diagnostics."""
    node_name = _qualified_name(node_type, path='PyTree node')
    if is_dataclass(node_type) and isinstance(auxiliary, tuple):
        node_fields = fields(node_type)
        static_fields = [field for field in node_fields if field.metadata.get('static', False)]
        # Equinox encodes module auxiliary data as (wrapper-fields slot, *static field values).
        # A focused test pins this layout so an upstream representation change fails explicitly.
        if len(auxiliary) == len(static_fields) + 1:
            return {
                'kind': 'tuple',
                'items': [
                    _encode_static(auxiliary[0], path=f'{node_name}.wrapper_fields'),
                    *[
                        _encode_static(value, path=f'{node_name}.{field.name}')
                        for field, value in zip(static_fields, auxiliary[1:], strict=True)
                    ],
                ],
            }
    return _encode_static(auxiliary, path=f'{node_name}.auxiliary')


def _tree_signature(treedef: jax.tree_util.PyTreeDef) -> dict[str, object]:
    """Encode node types and static auxiliary data from one JAX tree definition."""
    node_data = treedef.node_data()
    if node_data is None:
        return {'kind': 'leaf'}
    node_type, auxiliary = node_data
    return {
        'kind': 'node',
        'type': _qualified_name(node_type, path='PyTree node'),
        'auxiliary': _encode_tree_auxiliary(auxiliary, node_type),
        'children': [_tree_signature(child) for child in treedef.children()],
    }


def _path_text(path: tuple[object, ...]) -> str:
    """Render a JAX key path in a compact, user-facing form."""
    text = 'root'
    for key in path:
        if hasattr(key, 'name'):
            text += f'.{key.name}'
        elif hasattr(key, 'idx'):
            text += f'[{key.idx}]'
        elif hasattr(key, 'key'):
            text += f'[{key.key!r}]'
        else:
            text += f'[{key!r}]'
    return text


def _model_signature(
    model: object,
) -> tuple[dict[str, object], tuple[object, ...], jax.tree_util.PyTreeDef]:
    """Produce one ordered model signature and payload tuple from a single flatten traversal."""
    path_leaves, treedef = jax.tree_util.tree_flatten_with_path(model)
    entries = []
    payload = []
    for path, leaf in path_leaves:
        entry = {'path': _path_text(path), **_leaf_entry(leaf)}
        entries.append(entry)
        payload.append(leaf)
    return {'tree': _tree_signature(treedef), 'leaves': entries}, tuple(payload), treedef


def _first_difference(saved: object, current: object, path: str = 'root') -> str | None:
    """Return the first structural difference between canonical JSON-compatible values."""
    if type(saved) is not type(current):
        return f'{path}: saved {saved!r}, template {current!r}'
    if isinstance(saved, dict):
        for key in sorted(set(saved) | set(current), key=str):
            if key not in saved or key not in current:
                return f'{path}.{key}: key is missing from one signature'
            difference = _first_difference(saved[key], current[key], f'{path}.{key}')
            if difference is not None:
                return difference
        return None
    if isinstance(saved, list):
        if len(saved) != len(current):
            return f'{path}: saved length {len(saved)}, template length {len(current)}'
        for index, (left, right) in enumerate(zip(saved, current, strict=True)):
            difference = _first_difference(left, right, f'{path}[{index}]')
            if difference is not None:
                return difference
        return None
    if saved != current:
        return f'{path}: saved {saved!r}, template {current!r}'
    return None


def _without_payload_checksums(entries: object) -> object:
    """Return payload entries without their value-integrity metadata for schema comparisons."""
    if not isinstance(entries, list):
        return entries
    return [
        {key: value for key, value in entry.items() if key != 'sha256'}
        if isinstance(entry, dict)
        else entry
        for entry in entries
    ]


def _model_manifest(
    model: object,
) -> tuple[dict[str, object], tuple[object, ...], jax.tree_util.PyTreeDef]:
    """Build the complete version-1 model manifest and its matching payload tuple."""
    signature, payload, treedef = _model_signature(model)
    return (
        {
            'format': FORMAT_NAME,
            'format_version': FORMAT_VERSION,
            'kind': 'model',
            'jax_morph_version': _package_version(),
            'model_signature': signature,
            'payload': signature['leaves'],
        },
        payload,
        treedef,
    )


def save_model(path: str | os.PathLike[str], model: object) -> None:
    """Save a model's supported numeric PyTree leaves into a versioned artifact.

    The artifact stores numeric leaves only. Static model structure is recorded for validation and
    a caller-supplied template provides the executable Python objects at load time.

    Args:
        path: Destination artifact path.
        model: Model or other supported Equinox/JAX PyTree to serialize.

    Raises:
        PersistenceError: If the artifact cannot be written.
        TypeError: If the model contains unsupported payload leaves or static structure.
    """
    manifest, payload, _ = _model_manifest(model)
    _write_container(path, manifest, payload)


def load_model(path: str | os.PathLike[str], like: object) -> object:
    """Restore a model artifact's numeric leaves into an exactly matching ``like`` template.

    Args:
        path: Source artifact path.
        like: Matching live model template that supplies executable static objects.

    Returns:
        A copy of ``like`` with the artifact's numeric leaves restored.

    Raises:
        PersistenceError: If the artifact is malformed, incompatible, or does not match ``like``.
        TypeError: If ``like`` contains unsupported static structure.
    """
    with _read_container(path, expected_kind='model') as container:
        manifest = container.manifest
        saved_signature = manifest.get('model_signature')
        saved_entries = manifest.get('payload')
        if not isinstance(saved_signature, dict) or not isinstance(saved_entries, list):
            raise PersistenceError(
                'model artifact is missing its structure signature or payload entries'
            )
        if _without_payload_checksums(saved_entries) != saved_signature.get('leaves'):
            raise PersistenceError(
                'model artifact payload entries disagree with its structure signature'
            )
        current_signature, _, treedef = _model_signature(like)
        difference = _first_difference(saved_signature, current_signature, 'model_signature')
        if difference is not None:
            raise PersistenceError(f'model structure mismatch: {difference}')
        leaves = _read_payload(container, saved_entries)
    return jax.tree_util.tree_unflatten(treedef, leaves)


def _decode_spec_dtype(encoded: object) -> object:
    """Decode a representation-preserving declared StateFieldSpec dtype."""
    if not isinstance(encoded, dict):
        raise PersistenceError('declared dtype descriptor must be an object')
    kind = encoded.get('kind')
    if kind == 'none':
        return None
    if kind == 'python_builtin':
        builtins = {'bool': bool, 'int': int, 'float': float, 'complex': complex}
        name = encoded.get('name')
        if name in builtins:
            return builtins[name]
        raise PersistenceError(f'unsupported Python builtin dtype {name!r}')
    if kind == 'dtype_object':
        try:
            return jnp.dtype(encoded.get('value'))
        except TypeError as error:
            raise PersistenceError('invalid dtype-object descriptor') from error
    if kind == 'scalar_type':
        try:
            return jnp.dtype(encoded.get('value')).type
        except TypeError as error:
            raise PersistenceError('invalid scalar-type descriptor') from error
    if kind == 'jax_scalar_type':
        value = encoded.get('value')
        if value in _JAX_SCALAR_TYPES:
            return _JAX_SCALAR_TYPES[value]
        raise PersistenceError('invalid JAX scalar-type descriptor')
    if kind == 'string' and isinstance(encoded.get('value'), str):
        return encoded['value']
    raise PersistenceError(f'unsupported declared dtype kind {kind!r}')


def _decode_static(encoded: object) -> object:
    """Decode the non-executable subset of canonical static values used in state schemas."""
    if not isinstance(encoded, dict):
        raise PersistenceError('state schema value must be an object')
    kind = encoded.get('kind')
    if kind == 'none':
        return None
    if kind == 'bool' and type(encoded.get('value')) is bool:
        return encoded['value']
    if kind == 'int' and type(encoded.get('value')) is int:
        return encoded['value']
    if kind == 'float' and type(encoded.get('value')) in (int, float):
        value = float(encoded['value'])
        if math.isfinite(value):
            return value
    if kind == 'complex':
        real, imag = encoded.get('real'), encoded.get('imag')
        if type(real) in (int, float) and type(imag) in (int, float):
            value = complex(real, imag)
            if math.isfinite(value.real) and math.isfinite(value.imag):
                return value
    if kind == 'string' and isinstance(encoded.get('value'), str):
        return encoded['value']
    if kind == 'tuple' and isinstance(encoded.get('items'), list):
        return tuple(_decode_static(item) for item in encoded['items'])
    if kind == 'list' and isinstance(encoded.get('items'), list):
        return [_decode_static(item) for item in encoded['items']]
    if kind == 'dict' and isinstance(encoded.get('items'), list):
        return {_decode_static(key): _decode_static(value) for key, value in encoded['items']}
    raise PersistenceError(f'unsupported state schema static value kind {kind!r}')


def _spec_manifest(spec: StateFieldSpec) -> dict[str, object]:
    """Encode one complete state-field declaration for a self-describing state artifact."""
    return {
        'name': spec.name,
        'shape': list(spec.shape),
        'dtype': _encode_spec_dtype(spec.dtype, path=f'StateFieldSpec({spec.name!r}).dtype'),
        'heritable': spec.heritable,
        'default': _encode_static(spec.default, path=f'StateFieldSpec({spec.name!r}).default'),
        'scope': spec.scope,
    }


def _spec_from_manifest(encoded: object) -> StateFieldSpec:
    """Validate and decode one complete state-field declaration from an artifact manifest."""
    if not isinstance(encoded, dict):
        raise PersistenceError('state field spec must be an object')
    name, shape, heritable, scope = (
        encoded.get('name'),
        encoded.get('shape'),
        encoded.get('heritable'),
        encoded.get('scope'),
    )
    if not isinstance(name, str) or not name:
        raise PersistenceError('state field spec has an invalid name')
    if not isinstance(shape, list) or any(type(dim) is not int or dim < 0 for dim in shape):
        raise PersistenceError(f'state field {name!r} has an invalid declared shape')
    if type(heritable) is not bool or scope not in {'cell', 'global'}:
        raise PersistenceError(f'state field {name!r} has invalid heritable/scope metadata')
    return StateFieldSpec(
        name=name,
        shape=tuple(shape),
        dtype=_decode_spec_dtype(encoded.get('dtype')),
        heritable=heritable,
        default=_decode_static(encoded.get('default')),
        scope=scope,
    )


def _state_field_entries(state: BaseState) -> tuple[list[str], list[dict[str, object]]]:
    """Validate a generated state and return its ordered field names and payload entries."""
    if not isinstance(state, BaseState):
        raise TypeError('save_state requires a generated BaseState instance')
    if not isinstance(state.specs, Mapping):
        raise TypeError('state specs must be a mapping')
    names = list(state.specs)
    if len(names) != len(set(names)):
        raise TypeError('state field names must be unique')
    entries = []
    for name in names:
        spec = state.specs[name]
        if not isinstance(name, str) or not isinstance(spec, StateFieldSpec) or spec.name != name:
            raise TypeError(f'invalid state spec for field {name!r}')
        try:
            array = state[name]
        except AttributeError as error:
            raise TypeError(f'state is missing declared field {name!r}') from error
        descriptor = _leaf_entry(array)
        if descriptor['category'] != 'jax_array':
            raise TypeError(f'state field {name!r} must be a jax.Array')
        entries.append({'name': name, **descriptor})
    extra_arrays = [
        name
        for name, value in vars(state).items()
        if isinstance(value, jax.Array) and name not in state.specs
    ]
    if extra_arrays:
        raise TypeError(f'state has undeclared array fields {extra_arrays}')
    return names, entries


def _state_metadata(
    state: BaseState, *, require_unstacked: bool
) -> tuple[dict[str, object], tuple[object, ...]]:
    """Build the validated state schema section and matching name-ordered numeric payload."""
    names, entries = _state_field_entries(state)
    if require_unstacked and state.t.ndim != 0:
        raise ValueError(
            'save_state requires an unstacked state; use save_trajectory for histories'
        )
    if state.position.ndim != 2 or state.celltype.ndim != 2:
        raise TypeError('state position and celltype must be unstacked rank-2 arrays')
    capacity, n_space_dim = state.position.shape
    if state.celltype.shape[0] != capacity:
        raise TypeError('state celltype capacity does not match position capacity')
    payload = tuple(state[name] for name in names)
    return (
        {
            'state_schema_version': 1,
            'class_name': type(state).__name__,
            'specs': [_spec_manifest(state.specs[name]) for name in names],
            'field_names': names,
            'fields': entries,
            'capacity': capacity,
            'n_space_dim': n_space_dim,
            'n_types': state.celltype.shape[1],
            'space': _space_descriptor((state.displacement, state.shift)),
        },
        payload,
    )


def _state_manifest(state: BaseState) -> tuple[dict[str, object], tuple[object, ...]]:
    """Build the common manifest and payload for one unstacked state snapshot."""
    state_section, payload = _state_metadata(state, require_unstacked=True)
    return (
        {
            'format': FORMAT_NAME,
            'format_version': FORMAT_VERSION,
            'kind': 'state',
            'jax_morph_version': _package_version(),
            'state': state_section,
            'payload': state_section['fields'],
        },
        payload,
    )


def _state_from_values(
    section: object,
    entries: object,
    values: Sequence[object],
    *,
    space: object | None,
    n_frames: int | None = None,
) -> BaseState:
    """Rebuild a generated state class from already decoded name-ordered field values."""
    if not isinstance(section, dict) or section.get('state_schema_version') != 1:
        raise PersistenceError('unsupported or missing state schema version')
    class_name = section.get('class_name')
    field_names = section.get('field_names')
    specs_data = section.get('specs')
    fields_data = section.get('fields')
    if (
        not isinstance(class_name, str)
        or not isinstance(field_names, list)
        or not isinstance(specs_data, list)
        or not isinstance(fields_data, list)
        or entries != fields_data
    ):
        raise PersistenceError('state artifact has inconsistent schema and payload entries')
    if len(field_names) != len(set(field_names)) or any(
        not isinstance(name, str) for name in field_names
    ):
        raise PersistenceError('state artifact has duplicate or invalid field names')
    specs = [_spec_from_manifest(spec) for spec in specs_data]
    if [spec.name for spec in specs] != field_names:
        raise PersistenceError('state artifact field specs do not match field-name order')
    if len(fields_data) != len(field_names):
        raise PersistenceError('state artifact field count does not match its schema')
    for name, entry in zip(field_names, fields_data, strict=True):
        if (
            not isinstance(entry, dict)
            or entry.get('name') != name
            or entry.get('category') != 'jax_array'
        ):
            raise PersistenceError(
                f'state artifact has invalid payload metadata for field {name!r}'
            )
    try:
        resolved_space = _space_from_descriptor(section.get('space'), space=space)
    except (TypeError, ValueError) as error:
        raise PersistenceError(f'invalid saved space: {error}') from error
    if len(values) != len(field_names):
        raise PersistenceError('state payload value count does not match its schema')
    state_cls = _make_state_cls(class_name, {spec.name: spec for spec in specs})
    state = state_cls(space=resolved_space, **dict(zip(field_names, values, strict=True)))
    capacity, n_space_dim, n_types = (
        section.get('capacity'),
        section.get('n_space_dim'),
        section.get('n_types'),
    )
    if (
        type(capacity) is not int
        or type(n_space_dim) is not int
        or type(n_types) is not int
        or (n_frames is None and state.position.shape != (capacity, n_space_dim))
        or (n_frames is None and state.celltype.shape != (capacity, n_types))
        or (n_frames is not None and state.position.shape != (n_frames, capacity, n_space_dim))
        or (n_frames is not None and state.celltype.shape != (n_frames, capacity, n_types))
    ):
        raise PersistenceError(
            'state artifact redundant capacity/dimension metadata does not match fields'
        )
    return state


def _load_state_section(
    section: object, entries: object, payload: _ContainerPayload, *, space: object | None
) -> BaseState:
    """Read and rebuild a generated state class from a state-artifact payload."""
    if (
        not isinstance(section, dict)
        or not isinstance(section.get('fields'), list)
        or not isinstance(entries, list)
    ):
        raise PersistenceError('state artifact is missing field metadata')
    return _state_from_values(
        section,
        _without_payload_checksums(entries),
        _read_payload(payload, entries),
        space=space,
    )


def save_state(path: str | os.PathLike[str], state: BaseState) -> None:
    """Save one self-describing, unstacked generated state snapshot.

    Args:
        path: Destination artifact path.
        state: Unstacked generated state to serialize.

    Raises:
        PersistenceError: If the artifact cannot be written.
        TypeError: If the state or its schema is invalid or cannot be serialized.
        ValueError: If ``state`` is a stacked history.
    """
    manifest, payload = _state_manifest(state)
    _write_container(path, manifest, payload)


def load_state(path: str | os.PathLike[str], *, space: object | None = None) -> BaseState:
    """Load one self-describing state, reconstructing a built-in saved space when possible.

    Args:
        path: Source artifact path.
        space: Replacement ``(displacement, shift)`` pair for a custom saved space. Defaults to
            None.

    Returns:
        Reconstructed unstacked `BaseState` instance.

    Raises:
        PersistenceError: If the artifact is malformed, incompatible, or needs an unavailable
            custom space.
    """
    with _read_container(path, expected_kind='state') as payload:
        return _load_state_section(
            payload.manifest.get('state'),
            payload.manifest.get('payload'),
            payload,
            space=space,
        )


@dataclass(frozen=True)
class TrajectoryRecord:
    """A loaded complete trajectory and the metadata needed to interpret it.

    Attributes:
        history: Stacked complete `BaseState` history with leading frame axis ``n_steps + 1``.
        dt: Scalar JAX array containing the macro-step duration.
        provenance: JSON-safe mapping stored alongside the history.
    """

    history: BaseState
    dt: jax.Array
    provenance: Mapping[str, object]


def _normalise_dt(history: BaseState, dt: object) -> jax.Array:
    """Validate a real scalar step size and return the trajectory's scalar JAX representation."""
    if isinstance(dt, jax.Array):
        if dt.ndim != 0:
            raise ValueError('trajectory dt must be a scalar')
        value = dt
    elif type(dt) in (int, float):
        value = jnp.asarray(dt, dtype=history.t.dtype)
    else:
        raise ValueError('trajectory dt must be a finite real Python or JAX scalar')
    if jnp.issubdtype(value.dtype, jnp.bool_) or jnp.issubdtype(value.dtype, jnp.complexfloating):
        raise ValueError('trajectory dt must be a finite real scalar, not boolean or complex')
    if not bool(jax.device_get(jnp.isfinite(value))):
        raise ValueError('trajectory dt must be finite')
    return value


def _validate_history(
    history: BaseState, dt: object
) -> tuple[list[str], list[dict[str, object]], jax.Array]:
    """Validate the complete-history frame layout and its adjacent macro-step time axis."""
    names, entries = _state_field_entries(history)
    if history.t.ndim == 0:
        raise ValueError('save_trajectory requires a stacked history; use save_state for one state')
    if history.t.ndim != 1 or history.t.shape[0] == 0:
        raise ValueError('trajectory history.t must have one nonempty frame axis')
    if history.position.ndim != 3 or history.celltype.ndim != 3:
        raise ValueError(
            'trajectory position and celltype must have frame, capacity, and trailing axes'
        )
    n_frames, capacity, n_space_dim = history.position.shape
    if history.celltype.shape[0] != n_frames or history.celltype.shape[1] != capacity:
        raise ValueError('trajectory celltype frame/capacity axes do not match position')
    for name in names:
        array = history[name]
        spec = history.specs[name]
        minimum_ndim = 2 if spec.scope == 'cell' else 1
        if array.ndim < minimum_ndim or array.shape[0] != n_frames:
            raise ValueError(f'trajectory field {name!r} has an inconsistent frame axis')
        if spec.scope == 'cell' and array.shape[1] != capacity:
            raise ValueError(f'trajectory cell field {name!r} has an inconsistent capacity axis')
    value = _normalise_dt(history, dt)
    if n_frames > 1:
        dtype = np.dtype(history.t.dtype)
        tolerance = max(float(np.finfo(dtype).eps) * 16.0, 1e-7)
        if not bool(
            jax.device_get(jnp.allclose(jnp.diff(history.t), value, rtol=tolerance, atol=tolerance))
        ):
            raise ValueError('trajectory time increments are inconsistent with dt')
    return names, entries, value


def _trajectory_manifest(
    history: BaseState, dt: object, provenance: Mapping[str, object] | None
) -> tuple[dict[str, object], tuple[object, ...]]:
    """Build the complete-trajectory manifest and fixed history-fields-then-dt payload tuple."""
    names, fields_data, value = _validate_history(history, dt)
    if provenance is None:
        provenance_data: Mapping[str, object] = {}
    elif isinstance(provenance, Mapping):
        provenance_data = provenance
    else:
        raise TypeError('trajectory provenance must be a JSON-safe mapping')
    try:
        decoded_provenance = json.loads(_canonical_json(dict(provenance_data)).decode('utf-8'))
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise TypeError(f'trajectory provenance must be JSON-safe: {error}') from error
    if not isinstance(decoded_provenance, dict):
        raise TypeError('trajectory provenance must decode to a JSON object')
    n_frames = history.t.shape[0]
    state_section = {
        'state_schema_version': 1,
        'class_name': type(history).__name__,
        'specs': [_spec_manifest(history.specs[name]) for name in names],
        'field_names': names,
        'fields': fields_data,
        'capacity': history.position.shape[1],
        'n_space_dim': history.position.shape[2],
        'n_types': history.celltype.shape[2],
        'space': _space_descriptor((history.displacement, history.shift)),
    }
    dt_entry = {'name': 'dt', **_leaf_entry(value)}
    entries = [*fields_data, dt_entry]
    return (
        {
            'format': FORMAT_NAME,
            'format_version': FORMAT_VERSION,
            'kind': 'trajectory',
            'jax_morph_version': _package_version(),
            'state': state_section,
            'trajectory': {
                'history_contract_version': 1,
                'n_frames': n_frames,
                'n_steps': n_frames - 1,
                'dt': dt_entry,
                'provenance': decoded_provenance,
            },
            'payload': entries,
        },
        (*tuple(history[name] for name in names), value),
    )


def save_trajectory(
    path: str | os.PathLike[str],
    history: BaseState,
    dt: object,
    *,
    provenance: Mapping[str, object] | None = None,
) -> None:
    """Save one complete adjacent-macro-step history and its scalar ``dt`` metadata.

    Args:
        path: Destination artifact path.
        history: Complete stacked `BaseState` history with leading frame axis ``n_steps + 1``.
        dt: Finite real macro-step duration matching adjacent ``history.t`` frames.
        provenance: JSON-safe metadata to store with the trajectory. Defaults to None.

    Raises:
        PersistenceError: If the artifact cannot be written.
        TypeError: If history is invalid or provenance is not JSON-safe.
        ValueError: If history is not a complete stacked trajectory or ``dt`` is inconsistent.
    """
    manifest, payload = _trajectory_manifest(history, dt, provenance)
    _write_container(path, manifest, payload)


def load_trajectory(
    path: str | os.PathLike[str], *, space: object | None = None
) -> TrajectoryRecord:
    """Load a complete self-describing history, scalar JAX ``dt``, and provenance mapping.

    Args:
        path: Source artifact path.
        space: Replacement ``(displacement, shift)`` pair for a custom saved space. Defaults to
            None.

    Returns:
        `TrajectoryRecord` containing the stacked history, macro-step duration, and provenance.

    Raises:
        PersistenceError: If the artifact is malformed, incompatible, or needs an unavailable
            custom space.
    """
    with _read_container(path, expected_kind='trajectory') as payload:
        manifest = payload.manifest
        state_section = manifest.get('state')
        trajectory = manifest.get('trajectory')
        entries = manifest.get('payload')
        if (
            not isinstance(state_section, dict)
            or not isinstance(trajectory, dict)
            or not isinstance(entries, list)
        ):
            raise PersistenceError(
                'trajectory artifact is missing state, trajectory, or payload metadata'
            )
        n_frames, n_steps = trajectory.get('n_frames'), trajectory.get('n_steps')
        if (
            trajectory.get('history_contract_version') != 1
            or type(n_frames) is not int
            or n_frames < 1
            or n_steps != n_frames - 1
            or not isinstance(trajectory.get('provenance'), dict)
            or not isinstance(state_section.get('fields'), list)
        ):
            raise PersistenceError('trajectory artifact has invalid history metadata')
        field_entries = state_section['fields']
        dt_entry = trajectory.get('dt')
        if (
            _without_payload_checksums(entries) != [*field_entries, dt_entry]
            or not isinstance(dt_entry, dict)
            or dt_entry.get('name') != 'dt'
        ):
            raise PersistenceError('trajectory payload must contain history fields followed by dt')
        values = _read_payload(payload, entries)
        history = _state_from_values(
            state_section,
            field_entries,
            values[:-1],
            space=space,
            n_frames=n_frames,
        )
        dt = values[-1]
        if not isinstance(dt, jax.Array):
            raise PersistenceError('trajectory dt payload must be a JAX array')
        try:
            _validate_history(history, dt)
        except ValueError as error:
            raise PersistenceError(f'invalid loaded trajectory: {error}') from error
        provenance = json.loads(_canonical_json(trajectory['provenance']).decode('utf-8'))
    return TrajectoryRecord(history=history, dt=dt, provenance=provenance)
