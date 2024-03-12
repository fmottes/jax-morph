
import jax.numpy as np


def diff_avg_divrates(state):
    '''
    Difference of average growth rates between two cell types.
    '''
    
    #get index vectors for each type
    ctype_one = np.where(state.celltype==1,1,0)
    ctype_two = np.where(state.celltype==2,1,0)
    
    #get average divrate of each type 
    avdiv_one=np.sum(state.divrate*ctype_one)/np.sum(ctype_one)
    avdiv_two=np.sum(state.divrate*ctype_two)/np.sum(ctype_two)

    diff = avdiv_one-avdiv_two
    
    return diff


def diff_n_ctypes(state, relative=False):
    '''
    Difference in number of cells between two cell types.
    '''
    
    ctype_one = np.where(state.celltype==1,1,0).sum()
    ctype_two = np.where(state.celltype==2,1,0).sum()
    
    diff = ctype_one-ctype_two
    
    if relative:
        diff = diff/np.where(state.celltype!=0,1,0).sum()
    
    return diff

def cv_divrates(state):
    ''' 
    Coefficient of variation of division rates loss.
    '''
    return np.std(state.divrate)/np.mean(state.divrate)