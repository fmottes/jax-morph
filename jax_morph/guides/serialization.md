# Serialization

jax-morph persists three deliberately separate artifact kinds. All use the versioned `.jxm`
container: canonical JSON metadata plus numeric NPY leaves. Artifacts never use pickle, source
loading, or dynamic imports.

## Model checkpoints

Use a caller-created template to restore a model's numeric leaves:

```python
import jax_morph as jxm

jxm.save_model('model.jxm', model)
restored = jxm.load_model('model.jxm', build_model(dummy_parameters))
```

The template supplies classes, step ordering, static dimensions, field declarations, and
callables. The file supplies only ordinary-field `jax.Array` leaves and ordinary-field Python
`bool`, `int`, `float`, and `complex` leaves. Python scalar categories are restored exactly: they
remain filtered/static under Equinox, while array parameters remain arrays and optimizable.
`eqx.field(static=True)` values are structure, not checkpoint payload; they must match the template.
Other ordinary dynamic leaf types are rejected rather than copied from the template; declare genuine
structural objects and callables with `eqx.field(static=True)`.

## State snapshots

Save an unstacked final or intermediate state when a run should later continue:

```python
jxm.save_state('final.jxm', final_state)
state = jxm.load_state('final.jxm')
continued = jxm.simulate(model, state, n_steps=100, dt=0.1, key=new_key)
```

Snapshots contain their generated state schema, realized field dtypes/shapes, and boundary space,
so no model is needed to load them. Continuation always needs a caller-provided new key: an artifact
does not contain a live random-key cursor.

Free space and cached `periodic(box)` spaces restore automatically. A custom
`(displacement, shift)` pair is not executable data; provide it explicitly when loading:

```python
state = jxm.load_state('custom-space.jxm', space=my_space)
```

## Complete trajectories

Persist the complete history returned by `simulate(..., history=True)`, including `s_0`:

```python
jxm.save_trajectory('run.jxm', history, dt, provenance={'run': 'baseline'})
record = jxm.load_trajectory('run.jxm')
terms = jxm.trajectory_logp(candidate_model, record.history, record.dt)
```

`record.dt` is always a scalar JAX array. A JAX scalar preserves its dtype; a Python real scalar is
materialized using `history.t.dtype`. Provenance is JSON metadata, not a model compatibility gate:
a loaded history may intentionally be scored under another compatible candidate model.

Only complete adjacent macro-step histories are supported. Do not save a strided, subsampled, or
irregular observation sequence as a trajectory; changing `dt` cannot recreate omitted transition
traces.

## Compatibility and limits

Loaders check artifact kind/version, schema, numeric leaf category, shape, dtype, and payload EOF
before returning an object. A dtype that JAX would silently narrow, such as float64 with x64 disabled,
is rejected. Device placement, sharding, compiled executables, Python source, environments, training
state, compression, and streaming are intentionally outside this format.
