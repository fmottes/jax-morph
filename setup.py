from setuptools import setup, find_packages


INSTALL_REQUIRES = [
    'jax>=0.1.73',
    'jaxlib>=0.1.52',
    'jax-md>=0.1.28',
    'matplotlib',
    'equinox',
    'tqdm',
    'networkx',
]


setup(
    name='jax-morph',
    version='0.3',
    license='Apache 2.0',
    install_requires=INSTALL_REQUIRES,
    packages=find_packages(include=['jax_morph', 'jax_morph.*']),
)
