** ! WORK IN PROGRESS !**

# Jax-Morph

Jax-Morph is a Python library, mainly focused on simulating and optimizing clusters of cells in space. The library leverages the power of JAX for high-performance computations for efficient simulation and optimization workflows.

It is mainly built on top of [Equinox](https://github.com/patrick-kidger/equinox) and [JAX, M.D.](https://github.com/jax-md/jax-md) and designed to be flexible, expandable and adaptable to many scenarios that require physical simulations of interacting (active) particles. 

Main features:

- Simulation of cell clusters in space, with or without cell divisions
- Automatically differentiable, with optimization tools to learn cluster configurations
- Built on top of JAX for performance and scalability
- Thought to be **easy to exend and integrate** with other JAX-based libraries



# Installation

## Requirements (IMPORTANT):

### JAX and JAX, M.D.

There is currently a mismatch between the pip-released version of JAX MD and the latest version on GitHub. The released version uses some features of JAX that were deprecated, resulting in errors when used in conjunction with the latest JAX version. There are two ways around this at the moment:

**1. Latest JAX version, GitHub version of JAX-MD.**

Installed the latest JAX version following the [JAX Docs](https://github.com/google/jax?tab=readme-ov-file#installation). Then clone and install JAX-MD manually:
```bash
git clone https://github.com/google/jax-md
cd jax-md
pip install .
```
This is probably the most robust method.


**2. Older JAX version, Released JAX-MD version**

JAX version 0.4.23 seems to be the only one that do not cause conflicts with any of the other libraries involved, including JAX-MD. You can manually select the JAX version you prefer during install, just be mindful of specifying a compatible `jaxlib` version too.

For example, if you want to install wheels for JAX 0.4.23 and CUDA 12 you can run:
```
pip install jax==0.4.23 jaxlib==0.4.23+cuda12.cudnn89 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
After this you can install the released version of JAX-MD directly:
```pip install jax-md```

### Other libraries

These other required libraries should work in their latest version available on PPI:
- Equinox
- Matplotlib
- NetworkX
- Optax
- tqdm



## Install jax-morph

To install Jax-Morph, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/fmottes/jax-morph
    cd jax-morph
    ```

2. Install the package using pip:
    ```bash
    pip install .
    ```



## Usage

After installation, you can import and start using Jax-Morph in your projects.

```python
import jax_morph as jxm
```

## Quickstart

To get started quickly, check out the tutorial notebooks provided in the [tutorials](https://github.com/fmottes/jax-morph/tree/eqx/tutorials) directory. These notebooks cover basic usage and some advanced features of Jax-Morph.

_More coming soon_
