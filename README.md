# jax-morph
Simulation and optimization of clusters of cells growing in space. 

[Ramya demo colab](https://colab.research.google.com/drive/1man19YTDKaXqiV6WiKWCdWxLCiYAymTr?authuser=1#scrollTo=uKWBYfQXF-YI)
## Random notes:

- in order for `jax-morph` to work properly it needs to be installed by running

`pyhton setup.py install`

then it can be imported as a normal package anywhere.

- Usage of 32-bit precision floats instead of 64-bit ones leads to a considerable speedup (> 3x)
