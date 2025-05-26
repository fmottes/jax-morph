# Jax-Morph

**Note:** This repository contains the code to reproduce the results from [Deshpande, Mottes, et al. 2025](). For the latest version of Jax-Morph, please visit the [main repository](https://github.com/brenner-lab/jax-morph).

Jax-Morph is a Python library, mainly focused on simulating and optimizing clusters of cells in space. The library leverages the power of JAX for high-performance computations for efficient simulation and optimization workflows.

It is mainly built on top of [Equinox](https://github.com/patrick-kidger/equinox) and [JAX, M.D.](https://github.com/jax-md/jax-md) and designed to be flexible, expandable and adaptable to many scenarios that require physical simulations of interacting (active) particles, with a focus on simulation of biological systems at the cellular level.

Main features:

- Automatically differentiable, with optimization tools to learn cluster configurations
- Built on top of JAX for performance and scalability
- Thought to be **easy to exend and integrate** with other JAX-based libraries

---
# Installation

## Using Conda (Recommended)

The easiest way to install Jax-Morph with all dependencies is using the provided environment file:

1. Clone the repository:
    ```bash
    git clone https://github.com/fmottes/jax-morph
    cd jax-morph
    ```

2. Create and activate the conda environment:
    ```bash
    conda env create -f environment.yml
    conda activate jax-morph
    ```

This will automatically install:
- JAX with CUDA 12 support
- JAX-MD (latest GitHub version to ensure compatibility)
- All other required dependencies (Equinox, Diffrax, Optax, Matplotlib, tqdm)

## Alternative Installation

If you prefer to manage dependencies manually, you can install directly with pip:

```bash
pip install git+https://github.com/fmottes/jax-morph.git
```

**Note:** When installing manually, it is recommended to use the GitHub version of JAX-MD rather than the PyPI version:
```bash
pip install git+https://github.com/jax-md/jax-md.git
```

---

# Usage

After installation, you can import and start using Jax-Morph in your projects.

```python
import jax_morph as jxm
```

---
# Quickstart

For a detailed introduction, check out the tutorial notebooks provided in the [tutorials](./tutorials) directory:

## Jax-Morph Tutorials

These notebooks cover Jax-Morph usage from basic simulations to more advanced optimizations and simulations.

1. **[Simulation Basics](./tutorials/01%20-%20JAX-morph%20Simulation%20Basics.ipynb)** - Fundamentals of setting up and running cell cluster simulations
2. **[Gradients and Optimization](./tutorials/02%20-%20JAX-morph%20Gradients%20and%20Optimization.ipynb)** - Explore automatic differentiation and optimization capabilities in some simple models
3. **[Elongation](./tutorials/03%20-%20JAX-morph%20-%20Elongation.ipynb)** - Advanced example demonstrating cluster elongation dynamics

## Background: Introduction to Optimization

These notebooks provide background on optimization with differentiable simulations in general.

4. **[Deterministic Optimization](./tutorials/intro_to_optimization/Tutorial%2001%20-%20Optimization%20of%20deterministic%20simulations.ipynb)** - General introduction to optimization techniques for deterministic simulations
5. **[Stochastic Optimization](./tutorials/intro_to_optimization/Tutorial%2002%20-%20Optimization%20of%20stochastic%20simulations.ipynb)** - General introduction to optimization methods for stochastic systems

---
# Reproducing results from Deshpande, Mottes, et al. 2025

The code to reproduce the figures from [Deshpande, Mottes, et al. 2025]() is available in the [results-natcompsci-2025](./results-natcompsci-2025) directory. Each figure has its own subdirectory containing:

- **Training scripts** (`figX_train.py`) - Scripts to train the models
- **Model definitions** (`figX_istate_and_model.py`) - Initial states and model configurations  
- **Visualization notebooks** (`figX_visualizations.ipynb`) - Jupyter notebooks to generate the figures
- **Trained models** - Pre-trained model checkpoints (where applicable)

### Available figures:
- **[Figure 2](./results-natcompsci-2025/figure_2/)** - Spatial control of tissue growth (directional elongation)
- **[Figure 3](./results-natcompsci-2025/figure_3/)** - Chemical regulation of tissue homeostasis
- **[Figure 4](./results-natcompsci-2025/figure_4/)** - Mechano-chemical control of cell proliferation

To reproduce a specific figure, navigate to the corresponding directory and run the training script or open the visualization notebook.


---
# Reference

If you use Jax-Morph, please cite:

```
@article{deshpandemottes2025,
  title={Engineering morphogenesis of cell clusters with differentiable programming},
  author={Deshpande, Ramya and Mottes, Francesco and Vlad, Ariana-Dalia and Brenner, Michael P and dal Co, Alma},
  journal={arXiv preprint arXiv:2407.06295},
  year={2025}
}
```
