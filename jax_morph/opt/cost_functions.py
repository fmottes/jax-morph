import jax
import jax.numpy as np



def CellTypeImbalance(metric='number'):
    """
    Cost function for cell type imbalance.

    Metric must be either 'number' or 'entropy'.
    NOTE: 'number' option only works for 2 cell types.
    """

    # CELL TYPE IMBALANCE (ONLY 2 CELL TYPES)
    if 'number' == metric:

        def _cost(trajectory):

            cost = trajectory.celltype.sum(-2) @ np.asarray([1.,-1.])

            cost = np.abs(cost)

            cost = np.diff(cost)

            return cost


    # CELL TYPE DISTRIBUTION ENTROPY
    elif 'entropy' == metric:
        
        def _cost(trajectory):

            p = trajectory.celltype.sum(-2) / trajectory.celltype.sum(-2).sum(-1, keepdims=True)

            rel_entropy = -(p*np.log(p+1e-8)).sum(-1) / np.log(p.shape[-1])

            # minimize gap to max relative entropy
            cost = 1 - rel_entropy

            cost = np.diff(cost) * 10 # scale costs

            return cost
        
    else:
        raise ValueError(f'Metric must be either "number" or "entropy", not {type}.')

    return _cost




### ELONGATION
def MeanSquareY(*, nonsymm_penalty=1., realign=False):
    
    def _cost(trajectory):

        if realign:
            def _realign(pos):
                _, P = np.linalg.eigh(np.cov(pos.T))
                return pos @ P[:,::-1]
            pos = jax.vmap(_realign)(trajectory.position)

        else:
            pos = trajectory.position

        cost = (pos[:,:,1]**2).sum(-1)
        cost += nonsymm_penalty * np.abs(pos[:,:,0].sum(-1))

        cost = np.diff(cost)

        return cost
    
    return _cost




### V SHAPE
def VShape(*, cost_per_cell=3, rew_per_cell=1, nonsymm_penalty=.1, realign=False):
        
    def _cost(trajectory):

        if realign:
            def _realign(pos):
                _, P = np.linalg.eigh(np.cov(pos.T))
                return pos @ P[:,::-1]
            pos = jax.vmap(_realign)(trajectory.position)

        else:
            pos = trajectory.position

        mask = (pos[:,:,1]+1.5 > .5*np.abs(pos[:,:,0])) * (pos[:,:,1]+1.5 < 3.5+.5*np.abs(pos[:,:,0])) * (pos[:,:,1]>-.1)

        cost = np.sum(np.where(mask, -rew_per_cell, cost_per_cell) * trajectory.celltype.sum(-1), axis=-1)

        cost += nonsymm_penalty * np.abs(pos[:,:,0].sum(-1))

        cost = np.diff(cost)

        return cost
    
    return _cost



def CVDivrates(trajectory):
    cost = np.std(trajectory.division[-1, :])/np.mean(trajectory.division[-1, :])
    return cost




# def v_shape(trajectory):

#     def _state_cost(state):
#         pos = state.position
#         mask = (pos[:,1]+1.5 > .5*np.abs(pos[:,0])) * (pos[:,1]+1.5 < 3.5+.5*np.abs(pos[:,0])) * (pos[:,1]>-.1)

#         c = np.sum(np.where(mask, 0., 3.) * state.celltype.sum(-1))

#         return c


#     cost = jax.vmap(_state_cost)(trajectory)

#     cost += .1 * np.abs(trajectory.position[:,:,0].sum(-1))

#     cost = np.diff(cost)

#     return cost