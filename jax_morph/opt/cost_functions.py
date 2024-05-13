import jax
from functools import partial
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

            safe_p = np.where(p>0, p, 1)

            rel_entropy = -(p*np.log(safe_p)).sum(-1) / np.log(p.shape[-1])

            # maximize relative entropy
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
def VShape(*, cost_per_cell=1, rew_per_cell=3, nonsymm_penalty=.1, realign=False):
        
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



def CVDivrates():
    def _cost(trajectory):
        return np.std(trajectory.division[:, :], axis=1)/np.mean(trajectory.division[:, :], axis=1)
    return _cost


def PairwiseLoss(coeffs=None, kT_reward=.01, overlap_penalty=1.):                    
    def _cost(trajectory):
        
        displacement_fn = jax.vmap(jax.vmap(trajectory.displacement, (0, None)), (None, 0))
        displacements = jax.vmap(displacement_fn, (0,0))(trajectory.position, trajectory.position) + 1e-12
        distances = np.linalg.norm(displacements, axis=-1)

        # Pairwise types
        type_pair_f = jax.vmap(lambda x,y: x[:,None] @ y[None,:], (0,0))
        type_pair = jax.vmap(jax.vmap(type_pair_f, (-1, None)), (None, -1))(trajectory.celltype, trajectory.celltype)

        distance_pair = jax.vmap(lambda x: np.where(x, distances, 0.0), 0)(type_pair)
        
        costs = np.sum(distance_pair, (-2, -1))
        overlap = np.sum(np.where(distances < .85, 1., 0.0), (-2,-1))
        
        final_costs = np.sum(coeffs[:,:,None]*costs, (0, 1)) - kT_reward*trajectory.kT.sum() + overlap_penalty*overlap

        # Debugging printing.
        # print("Pairwise term: %s" % np.sum(coeffs[:,:,None]*costs, (0, 1)).mean())
        # print("kT term: %s" % (kT_reward*trajectory.kT.sum()).mean())
        # print("overlap term: %s" % (overlap_penalty*overlap).mean())
        
        return np.array([np.average(final_costs),])
    return _cost


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