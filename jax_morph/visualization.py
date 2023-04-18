import jax.numpy as np
import matplotlib.pyplot as plt



#set global properties of plots
plt.rcParams.update({'font.size': 15})



def draw_circles_ctype(state, ax=None, cm=plt.cm.coolwarm, **kwargs):
    
    if None == ax:
        ax = plt.axes()
    
    #only usable for two cell types
    color = cm(np.float32(state.celltype-1))

    for cell,radius,c in zip(state.position,state.radius,color):
        circle = plt.Circle(cell, radius=radius, color=c, alpha=.5, **kwargs)
        ax.add_patch(circle)
    
    
    ## calculate ax limits
    xmin = np.min(state.position[:,0])
    xmax = np.max(state.position[:,0])
    
    ymin = np.min(state.position[:,1])
    ymax = np.max(state.position[:,1])
    
    max_coord = max([xmax,ymax])+3
    min_coord = min([xmin,ymin])-3
    
    plt.xlim(min_coord,max_coord)
    plt.ylim(min_coord,max_coord)
    
    plt.xticks([])
    plt.yticks([])
    
    
    background_color = [56 / 256] * 3
    #ax.set_facecolor(background_color)    
    
    #ax.get_xaxis().set_visible(False)
    #ax.get_yaxis().set_visible(False)
    
    plt.gcf().patch.set_facecolor(background_color)
    plt.gcf().set_size_inches(6, 6)
    
    plt.tight_layout()
    
    return ax


def draw_circles_chem(state, chem=0, ax=None, cm=None, edges=False, cm_edges=plt.cm.coolwarm, **kwargs):
    
    if None == ax:
        ax = plt.axes()
    
    chemical = np.float32(state.chemical[:,chem])    
    chemical = (chemical-chemical.min())/(chemical.max()-chemical.min())
        
    #only usable for two cell types
    if cm is None:
        if 0 == chem:
            color = plt.cm.YlGn(chemical)
        elif 1 == chem:
            color = plt.cm.BuPu(chemical)
        else:
            color = plt.cm.coolwarm(chemical)
    else:
        color = cm(chemical)
        
    if edges:
        #only usable for two cell types
        ct_color = cm_edges(np.float32(state.celltype-1))

        for cell,radius,c,ctc in zip(state.position,state.radius,color,ct_color):
            circle = plt.Circle(cell, radius=radius, fc=c, ec=ctc, lw=2, alpha=.5, **kwargs)
            ax.add_patch(circle)
            
    else:
        for cell,radius,c in zip(state.position,state.radius,color):
            circle = plt.Circle(cell, radius=radius, fc=c, alpha=.5, **kwargs)
            ax.add_patch(circle)
            
    
    ## calculate ax limits
    xmin = np.min(state.position[:,0])
    xmax = np.max(state.position[:,0])
    
    ymin = np.min(state.position[:,1])
    ymax = np.max(state.position[:,1])
    
    max_coord = max([xmax,ymax])+3
    min_coord = min([xmin,ymin])-3
    
    plt.xlim(min_coord,max_coord)
    plt.ylim(min_coord,max_coord)
    
    plt.xticks([])
    plt.yticks([])
    
    
    background_color = [56 / 256] * 3
    #ax.set_facecolor(background_color)    
    
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
        
    plt.gcf().patch.set_facecolor(background_color)
    plt.gcf().set_size_inches(6, 6)
    
    plt.tight_layout()
    
    return ax


    
def draw_circles_divrate(state, ax=None, cm=plt.cm.coolwarm, edges=False, cm_edges=plt.cm.coolwarm, **kwargs):
    
    if None == ax:
        ax = plt.axes()
    
    divrate = np.float32(state.divrate)    
    divrate = (divrate-divrate.min())/(divrate.max()-divrate.min())
        
    #only usable for two cell types
    color = cm(divrate)
    
    if edges:
        #only usable for two cell types
        ct_color = cm_edges(np.float32(state.celltype-1))

        for cell,radius,c,ctc in zip(state.position,state.radius,color,ct_color):
            circle = plt.Circle(cell, radius=radius, fc=c, ec=ctc, lw=2, alpha=.5, **kwargs)
            ax.add_patch(circle)
            
    else:
        for cell,radius,c in zip(state.position,state.radius,color):
            circle = plt.Circle(cell, radius=radius, fc=c, alpha=.5, **kwargs)
            ax.add_patch(circle)
    
    
    
    
    ## calculate ax limits
    xmin = np.min(state.position[:,0])
    xmax = np.max(state.position[:,0])
    
    ymin = np.min(state.position[:,1])
    ymax = np.max(state.position[:,1])
    
    max_coord = max([xmax,ymax])+3
    min_coord = min([xmin,ymin])-3
    
    plt.xlim(min_coord,max_coord)
    plt.ylim(min_coord,max_coord)
    
    plt.xticks([])
    plt.yticks([])
    
    background_color = [56 / 256] * 3
    #ax.set_facecolor(background_color)    
    
    #ax.get_xaxis().set_visible(False)
    #ax.get_yaxis().set_visible(False)
        
    plt.gcf().patch.set_facecolor(background_color)
    plt.gcf().set_size_inches(6, 6)
    
    
def draw_circles(state, state_values, min_val = None, max_val = None, min_coord=None, max_coord=None, ax=None, cm=plt.cm.coolwarm, **kwargs):
    
    if None == ax:
        ax = plt.axes()
    
    state_values = np.float32(state_values)    
            
    if min_val == None:
        state_values = (state_values-state_values.min())/(state_values.max()-state_values.min())
    else:
        state_values = (state_values-min_val)/(max_val-min_val)

    #only usable for two cell types
    color = cm(state_values)
    for cell,radius,c in zip(state.position,state.radius,color):
        circle = plt.Circle(cell, radius=radius, fc=c, alpha=.5, **kwargs)
        ax.add_patch(circle)
    
    
    ## calculate ax limits
    xmin = np.min(state.position[:,0])
    xmax = np.max(state.position[:,0])
    
    ymin = np.min(state.position[:,1])
    ymax = np.max(state.position[:,1])
    
    if min_coord == None:
        max_coord = max([xmax,ymax])+3
        min_coord = min([xmin,ymin])-3
    
    ax.set_xlim(min_coord,max_coord)
    ax.set_ylim(min_coord,max_coord)
    
    ax.set_xticks([])
    ax.set_yticks([])
    
    background_color = [56 / 256] * 3
    #ax.set_facecolor(background_color)    
    
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
        
    #plt.gcf().patch.set_facecolor(background_color)
    #plt.gcf().set_size_inches(6, 6)
    
    #plt.tight_layout()
    
    return ax