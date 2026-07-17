# AGENTS.md

Working notes for developing **jax-morph**. Keep this file aligned with the repository as the
project evolves.

## What this is

Differentiable particle-based physics for proliferating cells and active matter, in JAX. The
library ships composable models and exposes optimization primitives (`trajectory_logp`,
`transition_logp`, and a pathwise-differentiable `simulate`); it owns **no** trainers or training
loops. Optimization examples belong in notebooks and guides, not in the package API.

- **Examples:** `examples/` - executed notebooks showcasing the current API (see its README).
- **Docs:** built from `docs/` with MkDocs Material; installed usage guides live under
  `jax_morph/guides/`.
- **Public package:** `jax_morph/` - the core, physics and control steps, serialization, and an
  optional matplotlib-backed `viz` API.

## Architecture

A small hardened **core** (`jax_morph/core/`) provides model-synthesized typed state, validated
field dataflow, the step contract, hybrid integration, stochastic trace replay/scoring,
serialization, and autodiff utilities. **Physics** and **control** steps compose on top solely by
reading and writing declared state fields.

A macro-step always runs phases in this order: `quasistatic` -> `dynamic` -> `discrete`.
Quasistatic and discrete steps update state sequentially in pipeline order. All dynamic steps read
the same post-quasistatic state and return `dt`-scaled sparse deltas; deltas targeting the same
field are added. Do not implement a dynamic step by returning a complete updated state.

Stochastic steps use one trace/replay contract: sample parameter-free noise and/or an action,
record declared ephemeral trace fields, replay the trace to produce the effect, and score the same
trace in `logp`. Trace fields reset every macro-step. `simulate(..., history=True)` records the
complete `s_0, ..., s_n` history; `trajectory_logp` scores it with live reconstructed state, while
`transition_logp` treats its input state as an observed conditioning boundary.

## Tech stack & tooling

- Python >=3.11, managed by **uv** (`.venv`, committed `uv.lock`).
- **Equinox** (PyTree modules), **JAX**, and **diffrax** (gene-network ODEs).
- **ruff** (lint + format), **pytest**, **MkDocs Material** + mkdocstrings.
- **matplotlib** is optional behind the `viz` extra; importing `jax_morph.viz` must continue to
  work without it, and rendering calls must raise an actionable install error.
- **optax** is notebook-only; do not make it a runtime dependency.
- No jax-md dependency in the core (free-space geometry, Morse, FIRE, Brownian implemented
  directly); the `neighbors` extra reserves a future sparse/neighbor-list backend.

## Commands

```bash
uv sync --all-extras --group dev --group docs --group notebook   # set up the environment
uv run ruff check .                              # lint
uv run ruff format --check .                     # verify formatting
uv run pytest -q                                 # default suite; x64 + debug_nans enabled
uv run pytest -m acceptance -q                   # heavy, main-gated acceptance test
uv run mkdocs build --strict                     # verify docs
```

Run `uv run ruff format .` to apply formatting. The default pytest invocation excludes the
`acceptance` marker through `pyproject.toml`. `.github/workflows/ci.yml` runs lint, format-check,
the default tests, a main-gated acceptance job, strict docs, and a separate base-install check that
verifies the visualization extra remains optional. `.github/workflows/docs.yml` builds pull
requests and deploys the docs from `main`.

**Managing dependencies:** always go through **uv**, never hand-edit `pyproject.toml`'s dependency
tables or `uv.lock`. Add a runtime dep with `uv add <pkg>`, a grouped dev/tooling dep with
`uv add --group <group> <pkg>` (e.g. `uv add --group dev matplotlib`), and an optional-extra dep
with `uv add --optional <extra> <pkg>`; remove with `uv remove [...] <pkg>`. This keeps
`pyproject.toml` and the committed `uv.lock` in sync. Commit both files together.

## Code conventions

- **Array type hints:** use `jax.Array` - never `np.ndarray`, `jnp.ndarray`, or `from jax import
  Array`.
- **Docstrings:** Google Python Style Guide. Type hints in function signatures only, not in
  docstring arg descriptions.
- **Quote style:** single quotes.
- **Plain ASCII in code and docstrings:** never use symbols like the Greek letter epsilon; write
  the name out (`epsilon`, `alpha`, `sigma`, `sqrt`, `->`, `<=`, ...). Keep source, comments, and
  docstrings ASCII-only.
- **Ruff:** `E501` (line length) and `E731` (lambda assignment) are ignored; pydocstyle (`D`) is
  enforced on library code (`jax_morph/`) but not on tests or notebooks. Config lives in
  `pyproject.toml`.
- **Equinox parameter convention:** store numeric parameters (epsilon, alpha, rates, coefficients)
  as **plain fields** (a Python scalar is static; a `jax.Array` is traced and optimizable).
  Reserve `eqx.field(static=True)` for shape-determining ints, scan lengths, field-spec dicts, and
  callables. All jit/grad goes through `eqx.filter_*`.
- **State-field coupling:** steps declare all inputs with `state_reads()`, persistent outputs with
  `state_writes()`, and ephemeral stochastic records with `trace_writes()`. Reuse the canonical
  base specs (`POSITION`, `RADIUS`, `CELLTYPE`, `ALIVE`, `TIME`) instead of redeclaring them.
- **Dynamic output:** return `state.deltas(...)` containing only this step's increment over `dt`.
  Untouched fields stay `None`; dynamic trace fields must have default zero.
- **Eligibility-trace naming:** when a stochastic step's action applies only to a subset of cells
  (those alive/eligible to draw at decision time), record that subset as a trace field named
  `{action}_eligible` (e.g. `divided` -> `divide_eligible`, `flipped` -> `flip_eligible`). `logp`
  masks the per-cell density by it, and the action prefix keeps trace fields unique per model.
  Prefer this over a bare `eligible` or re-reading `alive` at score time.
- **Gradient estimators:** continuous and reparameterized values may differentiate pathwise through
  `simulate`. For a sampled discrete choice, use either its straight-through surrogate or its
  score-function term, never both in one estimator.
- **Public API:** when adding or removing a public symbol, update the relevant package
  `__init__.py`, `__all__`, API docs, and public-API tests together.

## Dev workflow

- Behavioral changes are TDD: write a focused failing test, implement, then run the narrow test and
  the proportionate broader suite. Documentation/configuration-only changes should run their
  relevant validation command.
- **Commit after every successful subtask**, with a conventional-commit message.
- Do work on a feature branch, not `main`.
- **Usage guides are installed package resources.** Their canonical Markdown lives under
  `jax_morph/guides/`; the matching pages under `docs/` are relative symlinks for MkDocs. Edit the
  package copy. Read installed guides with `jax_morph.guides.guide()` and discover them with
  `jax_morph.guides.list_guides()`.
- **Serialization is non-executable.** Preserve the versioned JSON + NPY format; never introduce
  pickle, artifact-supplied imports, or executable payloads. Update serialization tests and the
  installed serialization guide when changing the format.
- **Example notebooks are committed as executed artifacts.** After changing one, re-execute it in
  place so its outputs stay in sync with the code:

  ```bash
  uv run jupyter nbconvert --to notebook --execute --inplace examples/01_core_walkthrough.ipynb
  ```
