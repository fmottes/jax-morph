import jax.numpy as np


#equinox automatically treats as static all non-array values
#cast floats/ints into arrays if the parameter needs a gradient
def _maybe_array(name, value, train_params):
    if train_params[name]:
        return np.array(value)
    else:
        return value


# Logistic function 
def logistic(x,gamma,k):
    return 1./(1.+np.exp(-gamma*(x-k)))

# Polynomial function
def polynomial(x, coeffs):
    return np.polyval(coeffs, x)