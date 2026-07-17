# Examples

Runnable walkthroughs of jax-morph. Notebooks are committed **with outputs** so they can be read
without running.

- **`01_core_walkthrough.ipynb`** - the Phase 1 core: building a `State` from a model, accessing
  and functionally updating fields, geometry and autodiff helpers, dataflow validation, `simulate`,
  conditional scoring with `transition_logp`, and end-to-end gradients.
- **`02_mechanics.ipynb`** - the Phase 2 mechanics: the `Morse` (energy, force, shared and
  per-cell couplings, virial pressure), `MechanicalRelaxation` (FIRE with implicit-diff equilibrium
  gradients), `BrownianDynamics` (free-diffusion MSD, score-function and pathwise gradients,
  aggregation), `VirialStress` (per-cell stress on a compressed cluster), and `SaturatingCellGrowth`
  (saturating growth at a per-cell rate, its gradient, and a growing + relaxing cluster) - with plots of the
  simulations and the gradients.
- **`03_potentials.ipynb`** - the interaction-potential zoo: overlaid energy and force curves for
  `Morse`, `SoftSphere`, `Hertzian`, `Harmonic`, and `LennardJones`; per-cell couplings (a per-cell
  state field in place of a per-type matrix); and a Brownian rollout contrasting a purely repulsive
  potential (spreads) with an adhesive one (condenses).
- **`04_physics_examples.ipynb`** - composing the physics layers into developing-cluster sims: a
  grow + relax + divide **proliferation cycle** (with the lineage tree recovered via
  `reconstruct_lineage`), **oriented vs isotropic cleavage** setting tissue shape (repulsive packing,
  so the division axis persists), and **contact-inhibited proliferation** where a `FreeScreenedDiffusion`
  signal senses crowding and throttles `Division` - closing with a check that the composed model is
  differentiable end to end.
- **`05_control_examples.ipynb`** - the Phase 3 control layer: four small models coupling a
  continuous-time **ODE controller** (a gene network) to the physics through shared state fields. An
  **active-matter** model where a `GeneNetworkConnectionist` reads `VirialStress` and throttles
  `ActiveBrownianDynamics2D` motility, producing motility-induced clustering; a **patterning** model
  where a `GeneNetworkMWC` reads a `FreeScreenedDiffusion` morphogen and drives differential growth into
  a lobe; a **homeostasis** model that *learns* a two-signal (activator / inhibitor) secretion program
  by the **score-function** estimator (REINFORCE through `trajectory_logp`, since the cell-count reward is
  non-differentiable) so the colony grows to a target number of cells and stops - it samples and
  detaches histories before differentiating the scorer; and **pathwise** training of a controller
  (mixing matrix frozen as a Python list) to a target gyration radius by `optax` through `simulate`,
  with the sampler deliberately inside the differentiated loss. These are the library's two
  optimization primitives, side by side.
- **`06_3d_visualization.ipynb`** - a compact three-dimensional model combining adhesive Brownian
  mechanics, saturating growth, and isotropic `Division`. It demonstrates physical icosphere
  snapshots, a fast marker animation of the growing population, shared 3-D limits and camera
  control, and identity-aware radius time series using the public `jxm.viz` API directly.

## Running

From a source checkout, install the notebook tools and all optional extras (including the
matplotlib visualization backend), then launch Jupyter with uv:

```bash
uv sync --all-extras --group dev --group notebook
```

```bash
uv run jupyter lab examples/01_core_walkthrough.ipynb
```

To re-execute a notebook in place:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace examples/01_core_walkthrough.ipynb
```
