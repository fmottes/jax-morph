from setuptools import setup, find_packages


INSTALL_REQUIRES = [
    "jax>=0.5.0",
    "jaxlib>=0.5.0",
    "jax-md>=0.2.25",
    "matplotlib==3.10.0",
    "equinox==0.11.11",
    "diffrax==0.6.2",
    "optax==0.2.4",
    "tqdm==4.67.1",
    "networkx>=3.4.2",
]


setup(
    name="jax-morph",
    version="0.3",
    description="A JAX-based library for simulating morphogenesis",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Francesco Mottes, Ramya Deshpande",
    author_email="fmottes@seas.harvard.edu, rdeshpande@seas.harvard.edu",
    url="https://github.com/fmottes/jax-morph",
    license="Apache 2.0",
    python_requires=">=3.8",
    install_requires=INSTALL_REQUIRES,
    packages=find_packages(include=["jax_morph", "jax_morph.*"]),
)
