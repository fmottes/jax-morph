import jax
import jax.numpy as np



### CELL TYPE IMBALANCE (ONLY 2 CELL TYPES)
def celltype_imbalance(trajectory):

    cost = trajectory.celltype.sum(-2) @ np.asarray([1.,-1.])

    cost = np.abs(cost)

    cost = np.diff(cost)

    return cost


### CELL TYPE DISTRIBUTION ENTROPY
def celltype_entropy(trajectory):

    p = trajectory.celltype.sum(-2) / trajectory.celltype.sum(-2).sum(-1, keepdims=True)

    #no minus sign since higher entropy is better
    cost = (p*np.log(p+1e-8)).sum(-1) 

    cost = np.diff(cost)

    return cost


### ELONGATION
def mean_square_pos_x(trajectory):

    cost = (trajectory.position[:,:,1]**2).sum(-1)

    cost += np.abs(trajectory.position[:,:,0].sum(-1))

    cost = np.diff(cost)

    return cost


### V SHAPE
def v_shape(trajectory):

    def _state_cost(state):
        pos = state.position
        mask = (pos[:,1]+1.5 > .5*np.abs(pos[:,0])) * (pos[:,1]+1.5 < 3.5+.5*np.abs(pos[:,0])) * (pos[:,1]>-.1)

        c = np.sum(np.where(mask, 0., 3.) * state.celltype.sum(-1))

        return c


    cost = jax.vmap(_state_cost)(trajectory)

    cost += .1 * np.abs(trajectory.position[:,:,0].sum(-1))

    cost = np.diff(cost)

    return cost