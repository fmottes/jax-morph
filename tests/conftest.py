import jax
import numpy as np
import pytest

jax.config.update('jax_enable_x64', True)
jax.config.update('jax_debug_nans', True)


@pytest.fixture
def key():
    return jax.random.PRNGKey(0)


def fd_grad(f, x, eps=1e-5):
    """Central finite-difference gradient of scalar f at array x."""
    x = np.asarray(x, dtype=np.float64)
    g = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'])
    for _ in it:
        i = it.multi_index
        xp = x.copy()
        xp[i] += eps
        xm = x.copy()
        xm[i] -= eps
        g[i] = (float(f(xp)) - float(f(xm))) / (2 * eps)
    return g
