import jax.numpy as np

# Logistic function 
def logistic(x,gamma,k):
    return 1./(1.+np.exp(-gamma*(x-k)))

# Polynomial function
def polynomial(x, coeffs):
    return np.polyval(coeffs, x)