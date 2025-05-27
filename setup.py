import os
from setuptools import setup, find_packages

# Get the long description from the README file
here = os.path.abspath(os.path.dirname(__file__))
try:
    with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = ""

INSTALL_REQUIRES = [
    "jax[cuda12]",
    "jax-md",
    "matplotlib",
    "equinox",
    "diffrax",
    "optax",
    "tqdm",
    "networkx",
]

setup(
    name="jax-morph",
    version="0.3",
    description="A JAX-based library for simulating morphogenesis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Francesco Mottes, Ramya Deshpande",
    author_email="fmottes@seas.harvard.edu, rdeshpande@seas.harvard.edu",
    url="https://github.com/fmottes/jax-morph",
    license="Apache 2.0",
    python_requires=">=3.8",
    install_requires=INSTALL_REQUIRES,
    packages=find_packages(include=["jax_morph", "jax_morph.*"]),
    include_package_data=True,
    zip_safe=False,
)
