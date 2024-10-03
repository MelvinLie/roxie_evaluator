import numpy as np
import gmsh

def get_quadrature_rule(order, element_type='Hex'):
    '''Get the Gaussian integration points, for a certain finite element.

    :param order:
        The order of the quadrature rule.

    :param element_type:
        The finite element type. Options are: Hex (default) for Hexahedral
        elements, and 'Quad', for quadrangles.

    :return:
        The integration points and weights.
    '''

    # initialize gmsh if not done already
    if not gmsh.isInitialized():
            gmsh.initialize()

    if element_type == 'Hex':
        gmsh_code = 5
    elif element_type == 'Quad':
        gmsh_code = 3

    q, w = gmsh.model.mesh.getIntegrationPoints(gmsh_code, "Gauss" + str(order))
    
    q.shape = (len(w), 3)

    return q, w
