import jax
import jax.numpy as np
import jax.tree_util as jtu
import equinox as eqx


# Logistic function
def logistic(x, gamma, k):
    return 1.0 / (1.0 + np.exp(-gamma * (x - k)))


# Polynomial function
def polynomial(x, coeffs):
    return np.polyval(coeffs, x)


def differentiable_clip(x, min=0.0, max=1.0):
    zero = x - jax.lax.stop_gradient(x)
    return zero + jax.lax.stop_gradient(np.clip(x, min, max))


# Rescaled algebraic sigmoid (better gradients - between 0 and 1)
def rescaled_algebraic_sigmoid(x):
    return 0.5 + (x / (2 * np.sqrt(1 + x**2)))


###-----------TRAJECTORY-----------------###


# converts a trajectory to a list of states
def traj_to_states(trajectory):

    def _to_state(i, traj):
        return jax.tree.map(lambda x: x[i], traj)

    _to_states = jax.jit(
        lambda traj: [_to_state(i, traj) for i in np.arange(traj.position.shape[0])]
    )

    return _to_states(trajectory)


###-----------PRINT GRADIENTS-----------------###


@jax.custom_vjp
def print_tangent(x):
    # This function will be used in the forward pass
    return x


def print_fwd(x):
    # Forward pass function for custom VJP
    return print_tangent(x), x


def print_bwd(x, g):
    # Backward pass function for custom VJP
    print(jtu.tree_flatten(g))
    print()

    return (g,)


print_tangent.defvjp(print_fwd, print_bwd)


###-----------NORMALIZE GRADIENTS-----------------###


@jax.custom_vjp
def normalize_grads(x):
    # This function will be used in the forward pass
    return x


def normalize_grads_fwd(x):
    # Forward pass function for custom VJP
    return normalize_grads(x), x


def normalize_grads_bwd(x, g):
    # Backward pass function for custom VJP

    # normalize gradients
    g = jtu.tree_map(lambda x: x / (np.linalg.norm(x) + 1e-20), g)

    return (g,)


normalize_grads.defvjp(normalize_grads_fwd, normalize_grads_bwd)


################################################
# Straight-Through Estimator for integer casting
################################################


@jax.custom_jvp
def to_int(x):
    return np.int32(x)


@to_int.defjvp
def to_int_jvp(primals, tangents):
    (x,) = primals
    (x_dot,) = tangents
    ans = x
    ans_dot = x_dot
    return ans, ans_dot


################################################
# Gradient discounting
################################################


@jax.custom_vjp
def discount_tangent(x, discount):
    return x


def discount_tangent_fwd(x, discount):
    return (
        discount_tangent(x, discount),
        discount,
    )  # Note that we return x, and t as residuals


def discount_tangent_bwd(res, g):
    discount = res  # res contains the forward pass residuals
    g_x = jax.tree_map(lambda x: discount * x, g)
    return g_x, 0.0


discount_tangent.defvjp(discount_tangent_fwd, discount_tangent_bwd)
