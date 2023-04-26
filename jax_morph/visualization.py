import jax.numpy as np
import matplotlib.pyplot as plt



#set global properties of plots
plt.rcParams.update({'font.size': 18})



def draw_circles_ctype(state, ax=None, cm=plt.cm.coolwarm, grid=False, **kwargs):
    
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
    

    #scale x and y in the same way
    ax.set_aspect('equal', adjustable='box')

    #white bg color for ax
    ax.set_facecolor([1,1,1])

    if grid:
        ax.grid(alpha=.2)
    else:
        #remove axis spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        plt.xticks([])
        plt.yticks([])


    background_color = [56 / 256] * 3        
    plt.gcf().patch.set_facecolor(background_color)
    plt.gcf().patch.set_alpha(0)

    plt.gcf().set_size_inches(8, 8)
    
    return plt.gcf(), ax


def draw_circles_chem(state, chem=0, colorbar=True, ax=None, cm=None, grid=False, edges=False, cm_edges=plt.cm.coolwarm, **kwargs):
    
    if None == ax:
        ax = plt.axes()
    
    chemical = np.float32(state.chemical[:,chem])    
    chemical = (chemical-chemical.min()+1e-20)/(chemical.max()-chemical.min()+1e-20)
        
    #only usable for two cell types
    if cm is None:
        if 0 == chem:
            cm = plt.cm.YlGn
        elif 1 == chem:
            cm = plt.cm.BuPu
        else:
            cm = plt.cm.coolwarm
        
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

    #show colorbar
    if colorbar:    
        sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(vmin=state.chemical[:,chem].min(), vmax=state.chemical[:,chem].max()))
        sm._A = []
        cbar = plt.colorbar(sm, shrink=0.7, alpha=.5) # rule of thumb
        cbar.set_label('Concentration Chem. '+str(chem), labelpad=20)

            
    
    ## calculate ax limits
    xmin = np.min(state.position[:,0])
    xmax = np.max(state.position[:,0])
    
    ymin = np.min(state.position[:,1])
    ymax = np.max(state.position[:,1])
    
    max_coord = max([xmax,ymax])+3
    min_coord = min([xmin,ymin])-3
    
    plt.xlim(min_coord,max_coord)
    plt.ylim(min_coord,max_coord)
    
    
    #scale x and y in the same way
    ax.set_aspect('equal', adjustable='box')

    #white bg color for ax
    ax.set_facecolor([1,1,1])

    if grid:
        ax.grid(alpha=.2)
    else:
        #remove axis spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        plt.xticks([])
        plt.yticks([])


    background_color = [56 / 256] * 3        
    plt.gcf().patch.set_facecolor(background_color)
    plt.gcf().patch.set_alpha(0)

    plt.gcf().set_size_inches(8, 8)
    
    return plt.gcf(), ax


    
def draw_circles_divrate(state, probability=False, colorbar=True, ax=None, cm=plt.cm.coolwarm, grid=False, edges=False, cm_edges=plt.cm.coolwarm, **kwargs):
    
    if None == ax:
        ax = plt.axes()
    
    if probability:
        divrate = state.divrate/(state.divrate.sum()+1e-20)
    else:
        divrate = np.float32(state.divrate)    
        divrate = (divrate-divrate.min()+1e-20)/(divrate.max()-divrate.min()+1e-20)
        
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


    
    #show colorbar
    if colorbar:    
        if probability:
            sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(vmin=divrate.min(), vmax=divrate.max()))
            sm._A = []
            cbar_text = 'Division Probability'
        else:
            sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(vmin=state.divrate.min(), vmax=state.divrate.max()))
            sm._A = []
            cbar_text = 'Division Propensity'

    
        cbar = plt.colorbar(sm, shrink=0.7, alpha=.5) # rule of thumb
        cbar.set_label(cbar_text, labelpad=20)
    
    ## calculate ax limits
    xmin = np.min(state.position[:,0])
    xmax = np.max(state.position[:,0])
    
    ymin = np.min(state.position[:,1])
    ymax = np.max(state.position[:,1])
    
    max_coord = max([xmax,ymax])+3
    min_coord = min([xmin,ymin])-3
    
    plt.xlim(min_coord,max_coord)
    plt.ylim(min_coord,max_coord)
    

    
    #scale x and y in the same way
    ax.set_aspect('equal', adjustable='box')

    #white bg color for ax
    ax.set_facecolor([1,1,1])

    if grid:
        ax.grid(alpha=.2)
    else:
        #remove axis spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        plt.xticks([])
        plt.yticks([])


    background_color = [56 / 256] * 3        
    plt.gcf().patch.set_facecolor(background_color)
    plt.gcf().patch.set_alpha(0)

    plt.gcf().set_size_inches(8, 8)

    return plt.gcf(), ax
    


    
def draw_circles(state, state_values, min_val = None, max_val = None, min_coord=None, max_coord=None, ax=None, cm=plt.cm.coolwarm, grid=False, **kwargs):
    
    if None == ax:
        ax = plt.axes()
    
    state_values = np.float32(state_values)    
            
    if min_val == None:
        state_values = (state_values-state_values.min()+1e-20)/(state_values.max()-state_values.min()+1e-20)
    else:
        state_values = (state_values-min_val+1e-20)/(max_val-min_val+1e-20)

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
    
    
    #scale x and y in the same way
    ax.set_aspect('equal', adjustable='box')

    #white bg color for ax
    ax.set_facecolor([1,1,1])

    if grid:
        ax.grid(alpha=.2)
    else:
        #remove axis spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        ax.set_xticks([])
        ax.set_yticks([])


    background_color = [56 / 256] * 3        
    plt.gcf().patch.set_facecolor(background_color)
    plt.gcf().patch.set_alpha(0)

    plt.gcf().set_size_inches(8, 8)

    return plt.gcf(), ax