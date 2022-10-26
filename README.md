# jax-morph
Simulation and optimization of clusters of cells growing in space.
[Ramya demo colab](https://colab.research.google.com/drive/1man19YTDKaXqiV6WiKWCdWxLCiYAymTr?authuser=1#scrollTo=uKWBYfQXF-YI)
## Random notes:

- in order for `jax-morph` to work properly it needs to be installed by running

`pyhton setup.py install`

then it can be imported as a normal package anywhere.

- A very minimal (for the moment) working example of how a simulation can be carried out can be found under Francesco/00 - Initial testing.ipynb. A more comprehensive (and comprehensible) guide will follow!

- Usage of 32-bit precision floats instead of 64-bit ones leads to a considerable speedup (> 3x)

- We can use Francesco and Ramya folders to test our different simulations while keeping it easy to access each other's code. The final notebooks with the simulations that we will put in the paper will be in the `notebooks` folder (and we'll delete Francesco and Ramya in the published version of course).
