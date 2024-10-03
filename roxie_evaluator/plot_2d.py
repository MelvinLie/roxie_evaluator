import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

def plot_mesh_2d(ax, nodes,
                 elements, element_codes,
                 colors=np.array([]), vertical_axis=1,
                 levels=100, limits=np.array([]),
                 cmap='jet', plot_edges=True):
    '''Plot a mesh in 2D.

    :param ax:
        A matplotlib axes object to plot into.

    :param nodes:
        The nodal coordinates.

    :param elements:
        The connectivity of the mesh.

    :param element_codes:
        The hypermesh codes of the finite elements.

    :param colors:
        An array specifying the colors of the nodes. If empty (default) only the mesh is shown.

    :param vertical_axis:
        Which is the vertical axis in the data. Default 1.

    :param levels:
        The number of levels for the contour plot if colors are given.

    :param limits:
        Limits for the colorbar of colors are given.

    :param cmap:
        A colormap.

    :param plot_edges:
        Set this flag if the edges of the mesh should be plotted. Default = True.
    '''

    # number of elements
    num_elements = elements.shape[0]

    # make empty table
    t = np.zeros((0,3),dtype=np.int32)

    # loop over all elements
    for i in range(num_elements):

        # switch between the different mesh types
        if element_codes[i] == 108:

            # these are 8 noded quadrilaterals
            
            # there will be 6  triangles
            
            this_t = np.zeros((6,3),dtype=np.int32)

            this_t[0,:] = np.array([elements[i,0]-1,elements[i,1]-1,elements[i,7]-1])
            this_t[1,:] = np.array([elements[i,1]-1,elements[i,2]-1,elements[i,3]-1])
            this_t[2,:] = np.array([elements[i,3]-1,elements[i,4]-1,elements[i,5]-1])
            this_t[3,:] = np.array([elements[i,5]-1,elements[i,6]-1,elements[i,7]-1])
            this_t[4,:] = np.array([elements[i,7]-1,elements[i,1]-1,elements[i,5]-1])
            this_t[5,:] = np.array([elements[i,1]-1,elements[i,3]-1,elements[i,5]-1])

            t = np.append(t,this_t,axis=0)

        if element_codes[i] == 106:

            # these are 6 noded triangular elements
            
            # there will be 4 triangles
            
            this_t = np.zeros((4,3),dtype=np.int32)

            this_t[0,:] = np.array([elements[i,0]-1,elements[i,1]-1,elements[i,5]-1])
            this_t[1,:] = np.array([elements[i,1]-1,elements[i,3]-1,elements[i,5]-1])
            this_t[2,:] = np.array([elements[i,1]-1,elements[i,2]-1,elements[i,3]-1])
            this_t[3,:] = np.array([elements[i,3]-1,elements[i,4]-1,elements[i,5]-1])


            t = np.append(t,this_t,axis=0)

    triang = mtri.Triangulation(nodes[:,0], nodes[:,vertical_axis], t)


    if len(colors) == nodes.shape[0]:

        # plot the meshgrid and colors
        if len(limits) == 0:
            tric = ax.tricontourf(triang, colors, cmap=cmap, levels = np.linspace(min(colors), max(colors), levels))
        else:
            tric = ax.tricontourf(triang, colors, cmap=cmap, levels = np.linspace(limits[0], limits[1], levels))

        if(plot_edges):
            ax.triplot(triang, lw=0.1, color='black')

    else:
        # plot only the meshgrid
        # tric = ax.triplot(triang,'k-',linewidth=1)
        tric = ax.tricontourf(triang, 0.6*np.ones((nodes.shape[0], )), cmap='Greys', levels = np.linspace(0., 1., 3))

    return tric