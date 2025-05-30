[![DOI](https://zenodo.org/badge/553123780.svg)](https://doi.org/10.5281/zenodo.15531405)


<div align="center">

# Jax-Morph

This work is dedicated to the memory of our dear friend and colleague, [Alma Dal Co](https://www.nature.com/articles/s41559-022-01978-7).

</div>

*This branch (and the associated release) contains the code to reproduce the results from [Deshpande, Mottes, et al. 2025](). A more general version of the library is under active development. In the future, please visit the [main branch](https://github.com/fmottes/jax-morph) for the latest version of Jax-Morph.*

Jax-Morph is a Python library, mainly focused on simulating and optimizing clusters of cells in space. The library leverages the power of JAX for high-performance computations for efficient simulation and optimization workflows.

It is mainly built on top of [Equinox](https://github.com/patrick-kidger/equinox) and [JAX, M.D.](https://github.com/jax-md/jax-md) and designed to be flexible, expandable and adaptable to many scenarios that require physical simulations of interacting (active) particles, with a focus on simulation of biological systems at the cellular level.

Main features:

- Automatically differentiable, with optimization tools to learn cluster configurations
- Built on top of JAX for performance and scalability
- Thought to be **easy to exend and integrate** with other JAX-based libraries

---
# Installation

You can install locally this **reproducibility version** of Jax-Morph with:

```bash
git clone -b paper-natcompsci-2025 https://github.com/fmottes/jax-morph
pip install -e jax-morph
```

**NOTE 1:** *This will automatically install the GPU version of JAX that packages the CUDA 12 toolkit.*

**NOTE 2:** *If you want to run the notebooks in Google Colab, you first need to install the package using the two commands above.*


---
# Usage

After installation, you can import and start using Jax-Morph in your projects.

```python
import jax_morph as jxm
```

See the [tutorial notebooks](./tutorials) and the next section for more details.

---
# Quickstart

For a detailed introduction, check out the tutorial notebooks provided in the [tutorials](./tutorials) directory.

**Jax-Morph Tutorials**

These notebooks cover Jax-Morph usage from basic simulations to more advanced optimizations and simulations.

1. **[Simulation Basics](./tutorials/01%20-%20JAX-morph%20Simulation%20Basics.ipynb)** - Fundamentals of setting up and running cell cluster simulations
2. **[Gradients and Optimization](./tutorials/02%20-%20JAX-morph%20Gradients%20and%20Optimization.ipynb)** - Explore automatic differentiation and optimization capabilities in some simple models
3. **[Elongation](./tutorials/03%20-%20JAX-morph%20-%20Elongation.ipynb)** - Advanced example demonstrating cluster elongation dynamics

**Background: Introduction to Optimization**

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

[![DOI](https://zenodo.org/badge/553123780.svg)](https://doi.org/10.5281/zenodo.15531405)


If you use Jax-Morph, please cite both the code DOI in the Zenodo badge above and the published paper linked in the release notes. 

This is the citation for the preprint version:

```
@article{deshpandemottes2025,
  title={Engineering morphogenesis of cell clusters with differentiable programming},
  author={Deshpande, Ramya and Mottes, Francesco and Vlad, Ariana-Dalia and Brenner, Michael P and Dal Co, Alma},
  journal={arXiv preprint arXiv:2407.06295},
  year={2025}
}
```
