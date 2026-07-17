---
hide:
  - toc
---

<p align="center">
  <img src="_static/images/jax-morph-wordmark-white.png" alt="jax-morph" width="720">
</p>

# Getting started

Differentiable particle-based physics for proliferating cells and active matter, built with JAX and Equinox.

jax-morph provides composable simulation **models** and exposes the primitives an optimizer needs
(`trajectory_logp` for complete stochastic histories, `transition_logp` for conditional
transitions, and a pathwise-differentiable `simulate`). It does **not** ship training loops - see
the optimization guides for user-written examples.

## Key features

*   **Differentiable simulation**: differentiate continuous dynamics and model parameters end to
    end with JAX and Equinox.
*   **Composable model steps**: combine quasistatic constraints, dynamic increments, and discrete
    events under one validated state contract.
*   **Two optimization estimators**: put `simulate` inside a pathwise objective, or sample first and
    score detached stochastic trajectories with `trajectory_logp` for REINFORCE.
*   **Cell-cluster physics and control**: assemble mechanics, diffusion, growth, division, death,
    and continuous-time controllers without adopting a library-owned training loop.

## Installation

- **pip**

```bash
pip install jax-morph
```

- **uv**

```bash
uv add jax-morph
```

### Visualization

Static rendering, animation, and per-cell time-series plots use the optional matplotlib extra:

- **pip**

```bash
pip install 'jax-morph[viz]'
```

- **uv**

```bash
uv add 'jax-morph[viz]'
```

The base installation remains matplotlib-free. Importing `jax_morph.viz` is safe without the
extra; only a rendering call requires it.

Continue with [Basic Usage](basic-usage.md), then use the [example notebooks](example-notebooks/README.md)
for complete runnable workflows.
