> [!IMPORTANT]
> The legacy code to reproduce the results in
> [*Engineering morphogenesis of cell clusters with differentiable programming*](https://doi.org/10.1038/s43588-025-00851-4)
> is available on the [`paper-natcompsci-2025` branch](https://github.com/fmottes/jax-morph/tree/paper-natcompsci-2025). 
> This old version should only be used to reproduce the results of the paper. For any other purpose, please use the current code on `main`. Note that the current library API is **not backward-compatible** with the legacy version.



<p align="center">
  <img src="docs/_static/images/jax-morph-wordmark-white.png" alt="jax-morph" width="720">
</p>

Differentiable particle-based physics for proliferating cells and active matter, built with JAX and Equinox.

- **Differentiable end-to-end** — pathwise gradients through the continuous physics, score-function gradients through discrete events.
- **Composable physics steps** — mechanics, diffusion, Brownian and active-Brownian dynamics, growth, division, and death.
- **Principled time-stepping** — steps compose into one macro-step by Lie-Trotter operator splitting: each advances the shared state over `dt`, giving a consistent, first-order-accurate integrator of the coupled dynamics.
- **Continuous-time dynamics and control** — neural-ODE and gene-network controllers that model cell decision making from local cues.
- **Core abstractions built for easy extension** — add your own physics steps to the pipeline. Guides provided with the library for coding agents reference.
- **JAX and Equinox native** — `jit`, `vmap`, Equinox filtered transformations and neural-network modules work throughout the pipeline; GPU support out of the box.

See [Key Concepts](https://fmottes.github.io/jax-morph/concepts.html) and usage guides for more.

---
## Installation

jax-morph requires Python 3.11 or later.

```bash
pip install jax-morph
```

Or with uv:

```bash
uv add jax-morph
```

The visualization API is an optional matplotlib-backed extra. Install it when using
`jxm.viz.draw`, `jxm.viz.animate`, or `jxm.viz.plot_timeseries`:

```bash
pip install 'jax-morph[viz]'
```

Or with uv:

```bash
uv add 'jax-morph[viz]'
```

The base package does not install matplotlib; `import jax_morph.viz` remains available, and a
rendering call explains how to add the extra if it is missing.

---
## Quickstart

This model seeds a single cell that diffuses under a pairwise interaction and stochastically divides:

```python
import jax

import jax_morph as jxm
from jax_morph.physics import BrownianDynamics, Division, SoftSphere

# Cells diffuses under a soft-sphere interaction and divide stochastically
model = jxm.Model([
    BrownianDynamics(SoftSphere(), n_space_dim=2, kT=0.05),
    Division(n_space_dim=2),
])

# Build state class and initialize with a single cell
StateClass = jxm.build_state_from_model(model)
state = StateClass.init_empty(capacity=32, n_space_dim=2, n_types=1)
state = state.update(
    alive=state.alive.at[0].set(True),
    radius=state.radius.at[0].set(0.5),
    celltype=state.celltype.at[0, 0].set(1.0),
    division_rate=state.division_rate.at[0].set(1.0),
)

# Simulate
sim_key = jax.random.PRNGKey(0)
final_state = jxm.simulate(model, state, n_steps=20, dt=0.1, key=sim_key)
```

See the [documentation](https://fmottes.github.io/jax-morph/index.html) and [example notebooks](https://fmottes.github.io/jax-morph/example-notebooks/index.html) for more.

---
## Installed usage guides

Usage guides ship with the library so they remain available with a PyPI installation and match the
installed API version:

```python
import jax_morph as jxm

jxm.guides.list_guides()
jxm.guides.guide('extending')
jxm.guides.guide('optimization/pathwise')
```

---
## Reference

If you use JAX-Morph, please cite:

```
@article{deshpandemottes2025,
  title={Engineering morphogenesis of cell clusters with differentiable programming},
  author={Deshpande, Ramya and Mottes, Francesco and Vlad, Ariana-Dalia and Brenner, Michael P and Dal Co, Alma},
  journal   = {Nature Computational Science},
  year      = {2025},
  doi       = {10.1038/s43588-025-00851-4},
  url       = {https://doi.org/10.1038/s43588-025-00851-4}
}
```

The published paper can be read [here](https://rdcu.be/eAyqo).
