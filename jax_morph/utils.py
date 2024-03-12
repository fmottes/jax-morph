import jax
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


def differentiable_clip(x, min=0., max=1.):
    zero = x - jax.lax.stop_gradient(x)
    return zero + jax.lax.stop_gradient(np.clip(x, min, max))


################################################
# Straight-Through Estimator for integer casting
################################################

@jax.custom_jvp
def to_int(x):
    return np.int32(x)

@to_int.defjvp
def to_int_jvp(primals, tangents):
    x, = primals
    x_dot, = tangents
    ans = x
    ans_dot = x_dot
    return ans, ans_dot