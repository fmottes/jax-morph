# jax-morph
Simulation and optimization of clusters of cells growing in space. 

## Random notes:

- in order for `jax-morph` to work properly it needs to be installed by running

`pip install -e .`

This will install the package in developer mode (all of the changes are immediately available without rebuilding). Then it can be imported as a normal package anywhere in the environment.

- Usage of 32-bit precision floats instead of 64-bit ones leads to a considerable speedup (> 3x)
