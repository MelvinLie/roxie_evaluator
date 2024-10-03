import numpy as np
import matplotlib.pyplot as plt
import gmsh

''' ===================================
       Hexahedral finite elements

        Author: Melvin Liebsch
    Email: melvin.liebsch@cern.ch
====================================='''

def hex_nodal_coordinates(num_nodes):
    '''For a hexahedral element, get all the nodal coordinates (local).
    We follow the definitions of "Numerical Solution of Partial Differential
    Equations in Science and Engineering". See page 155.
    But we use the node numbering scheme of gmsh. 
    See https://gmsh.info/doc/texinfo/gmsh.html#index-Mesh_002eElementOrder

    :param num_nodes:
        The number of nodes.

    :return:
        A numpy array of dimension (num_nodes x 3), with the nodal coordinates
        (local) in the rows.
    '''

    if num_nodes == 8:
        ret_array = np.array([[-1.0, -1.0, -1.0],
                              [1.0, -1.0, -1.0],
                              [1.0, 1.0, -1.0],
                              [-1.0, 1.0, -1.0],
                              [-1.0, -1.0, 1.0],
                              [1.0, -1.0, 1.0],
                              [1.0, 1.0, 1.0],
                              [-1.0, 1.0, 1.0]])

    elif num_nodes == 20:
        ret_array = np.array([[-1.0, -1.0, -1.0],
                              [1.0, -1.0, -1.0],
                              [1.0, 1.0, -1.0],
                              [-1.0, 1.0, -1.0],
                              [-1.0, -1.0, 1.0],
                              [1.0, -1.0, 1.0],
                              [1.0, 1.0, 1.0],
                              [-1.0, 1.0, 1.0],
                              [0.0, -1.0, -1.0],
                              [-1.0, 0.0, -1.0],
                              [-1.0, -1.0, 0.0],
                              [1.0, 0.0, -1.0],
                              [1.0, -1.0, 0.0],
                              [0.0, 1.0, -1.0],
                              [1.0, 1.0, 0.0],
                              [-1.0, 1.0, 0.0],
                              [0.0, -1.0, 1.0],
                              [-1.0, 0.0, 1.0],
                              [1.0, 0.0, 1.0],
                              [0.0, 1.0, 1.0]])
        
    elif num_nodes == 32:
        ret_array = np.array([[-1.0, -1.0, -1.0],
                              [1.0, -1.0, -1.0],
                              [1.0, 1.0, -1.0],
                              [-1.0, 1.0, -1.0],
                              [-1.0, -1.0, 1.0],
                              [1.0, -1.0, 1.0],
                              [1.0, 1.0, 1.0],
                              [-1.0, 1.0, 1.0],
                              [-1/3, -1.0, -1.0],
                              [1/3, -1.0, -1.0],
                              [-1.0, -1/3, -1.0],
                              [-1.0, 1/3, -1.0],
                              [-1.0, -1.0, -1/3],
                              [-1.0, -1.0, 1/3],  
                              [1.0, -1/3, -1.0],
                              [1.0, 1/3, -1.0], 
                              [1.0, -1.0, -1/3],
                              [1.0, -1.0, 1/3],
                              [1/3, 1.0, -1.0],
                              [-1/3, 1.0, -1.0],    
                              [1.0, 1.0, -1/3],
                              [1.0, 1.0, 1/3],          
                              [-1.0, 1.0, -1/3],
                              [-1.0, 1.0, 1/3], 
                              [-1/3, -1.0, 1.0],
                              [1/3, -1.0, 1.0],        
                              [-1.0, -1/3, 1.0],
                              [-1.0, 1/3, 1.0],    
                              [1.0, -1/3, 1.0],
                              [1.0, 1/3, 1.0],
                              [1/3, 1.0, 1.0],
                              [-1/3, 1.0, 1.0]])

    else:
        print('The Hexahedral element with {} nodes not found!'.format(num_nodes))

        ret_array = np.zeros((0, 3))

    # ret_array *= 2.0
    # ret_array[:, 0] -= 1.0
    # ret_array[:, 1] -= 1.0
    # ret_array[:, 2] -= 1.0

    return ret_array

def hex_node_types(num_nodes):
    '''For a hexahedral element, get all the node types.
    Type   |      Meaning
    ========================
       1   |     corner
       2   |  edge || w
       3   |  edge || u
       4   |  edge || v

    We follow the definitions of "Numerical Solution of Partial Differential
    Equations in Science and Engineering". See page 155.
    Notice: The order of the shape functions is different to what is given
    in the book for the 32 node element.
    But we use the node numbering scheme of gmsh. 
    See https://gmsh.info/doc/texinfo/gmsh.html#index-Mesh_002eElementOrder


    :param num_nodes:
        The number of nodes.

    :return:
        A numpy array of integers. These are the node types.
    '''

    if num_nodes == 8:
        ret_array = np.array([1, 1, 1, 1, 1, 1, 1, 1], dtype=np.int32)

    elif num_nodes == 20:
        ret_array = np.array([1, 1, 1, 1, 1, 1, 1, 1,
                              3, 4, 2,
                              4, 2, 3,
                              2, 2, 
                              3, 4, 4, 3], dtype=np.int32)

    elif num_nodes == 32:
        ret_array = np.array([1, 1, 1, 1, 1, 1, 1, 1,
                              3, 3, 4, 4, 2, 2,
                              4, 4, 2, 2, 3, 3,
                              2, 2, 2, 2,
                              3, 3, 4, 4, 4, 4, 3, 3], dtype=np.int32)


    else:
        print('The Hexahedral element with {} nodes does not exist!'.format(num_nodes))

        ret_array = np.zeros((0,))

    return ret_array

def eval_shape_hexahedron(points, order):
    '''Evaluate the shape functions of a Hexahedral finite element.

    :param points:
        A numpy array of dimension (M, 3) with the local coordinates (u,v,w)
        in (-1, 1)^3 in each row.

    :param order:
        The finite element order. Needs to be between 1 and 3.

    :return:
        An (M x num_nodes) numpy array of shape function evaluations.
    '''

    # we make the number of nodes dependent on the element order
    # we currently use incomplete elements only
    if order < 1:
        print('FEM order {} is smaller than 1!'.format(order))
    elif order == 1:
        el_nodes = 8
    elif order == 2:
        el_nodes = 20
    elif order == 3:
        el_nodes = 32
    else:
        print('FEM order {} is larger than 3!'.format(order))

    # get the local nodal coordinates
    nodes = hex_nodal_coordinates(el_nodes)

    # get the node types
    types = hex_node_types(el_nodes)

    # get the number of evaluation points
    M = points.shape[0]

    # the number of nodes
    num_nodes = nodes.shape[0]

    # allocate space for the return array
    ret_vals = np.zeros((M, num_nodes))

    # loop over the basis functions
    for i in range(num_nodes):
        # there are only four types

        # corner nodes
        if types[i] == 1:

            # Hex8 Element
            if num_nodes == 8:
                ret_vals[:, i] = 0.125*( (1 + nodes[i, 0]*points[:, 0])
                                        *(1 + nodes[i, 1]*points[:, 1])
                                        *(1 + nodes[i, 2]*points[:, 2]) )

            # Hex20 Element
            elif num_nodes == 20:
                ret_vals[:, i] = 0.125*( (1 + nodes[i, 0]*points[:, 0])
                                        *(1 + nodes[i, 1]*points[:, 1])
                                        *(1 + nodes[i, 2]*points[:, 2])
                                        *(nodes[i, 0]*points[:, 0]
                                            + nodes[i, 1]*points[:, 1]
                                            + nodes[i, 2]*points[:, 2] - 2.0) )

            # Hex32 Element
            elif num_nodes == 32:
                ret_vals[:, i] = 1.0/64*( (1 + nodes[i, 0]*points[:, 0])
                                         *(1 + nodes[i, 1]*points[:, 1])
                                         *(1 + nodes[i, 2]*points[:, 2])
                                         *(9*(points[:, 0]**2
                                            + points[:, 1]**2
                                            + points[:, 2]**2) - 19.0) )

            else:
                print('Hexahedral element with {} not found!'.format(num_nodes))

        # corner edge || w
        if types[i] == 2:

            # Hex20 Element
            if num_nodes == 20:
                ret_vals[:, i] = 0.25*( (1 - points[:, 2]**2)
                                        *(1 + nodes[i, 0]*points[:, 0])
                                        *(1 + nodes[i, 1]*points[:, 1] ) )

            # Hex32 Element
            elif num_nodes == 32:
                ret_vals[:, i] = 9.0/64*( (1 - points[:, 2]**2)
                                        *(1 + 9.0*nodes[i, 2]*points[:, 2])
                                        *(1 + nodes[i, 0]*points[:, 0])
                                        *(1 + nodes[i, 1]*points[:, 1]) )

            else:
                print('Hexahedral element with {} not found!'.format(num_nodes))


        # corner edge || u
        if types[i] == 3:

            # Hex20 Element
            if num_nodes == 20:
                ret_vals[:, i] = 0.25*( (1 - points[:, 0]**2)
                                        *(1 + nodes[i, 2]*points[:, 2])
                                        *(1 + nodes[i, 1]*points[:, 1] ) )

            # Hex32 Element
            elif num_nodes == 32:
                ret_vals[:, i] = 9.0/64*( (1 - points[:, 0]**2)
                                        *(1 + 9.0*nodes[i, 0]*points[:, 0])
                                        *(1 + nodes[i, 1]*points[:, 1])
                                        *(1 + nodes[i, 2]*points[:, 2]) )

            else:
                print('Hexahedral element with {} not found!'.format(num_nodes))

        # corner edge || v
        if types[i] == 4:

            # Hex20 Element
            if num_nodes == 20:
                ret_vals[:, i] = 0.25*( (1 - points[:, 1]**2)
                                        *(1 + nodes[i, 2]*points[:, 2])
                                        *(1 + nodes[i, 0]*points[:, 0]) )

            # Hex32 Element
            elif num_nodes == 32:
                ret_vals[:, i] = 9.0/64*( (1 - points[:, 1]**2)
                                        *(1 + 9.0*nodes[i, 1]*points[:, 1])
                                        *(1 + nodes[i, 2]*points[:, 2])
                                        *(1 + nodes[i, 0]*points[:, 0]) )

            else:
                print('Hexahedral element with {} not found!'.format(num_nodes))


    return ret_vals


def eval_gradient_hexahedron(points, order):
    '''Evaluate the gradient of the shape functions of a Hexahedral finite
    element.

    :param points:
        A numpy array of dimension (M, 3) with the local coordinates (u,v,w)
        in (-1, 1)^3 in each row.

    :param order:
        The finite element order. Needs to be between 1 and 3.

    :return:
        An (M x num_nodes x 3) numpy array of shape function gradient
        evaluations.
    '''

    # we make the number of nodes dependent on the element order
    # we currently use incomplete elements only
    if order < 1:
        print('FEM order {} is smaller than 1!'.format(order))
    elif order == 1:
        el_nodes = 8
    elif order == 2:
        el_nodes = 20
    elif order == 3:
        el_nodes = 32
    else:
        print('FEM order {} is larger than 3!'.format(order))

    # get the local nodal coordinates
    nodes = hex_nodal_coordinates(el_nodes)

    # get the node types
    types = hex_node_types(el_nodes)

    # get the number of evaluation points
    M = points.shape[0]

    # the number of nodes
    num_nodes = nodes.shape[0]

    # allocate space for the return array
    ret_vals = np.zeros((M, num_nodes, 3))

    # get the local coordinates
    u = points[:, 0]
    v = points[:, 1]
    w = points[:, 2]

    # loop over the basis functions
    for i in range(num_nodes):

        # this is for a cleaner code
        u_i = nodes[i, 0]
        v_i = nodes[i, 1]
        w_i = nodes[i, 2]

        # corner nodes
        if types[i] == 1:

            # Hex8 Element
            if num_nodes == 8:
                ret_vals[:, i, 0] = 0.125*u_i*(v*v_i + 1)*(w*w_i + 1)
                ret_vals[:, i, 1] = 0.125*v_i*(u*u_i + 1)*(w*w_i + 1)
                ret_vals[:, i, 2] = 0.125*w_i*(u*u_i + 1)*(v*v_i + 1)

            # Hex20 Element
            elif num_nodes == 20:
                ret_vals[:, i, 0] = 0.125*(u_i*(v*v_i + 1)
                                                *(w*w_i + 1)
                                                *(2*u*u_i
                                                + v*v_i
                                                + w*w_i - 1.0))

                ret_vals[:, i, 1] = 0.125*(v_i*(u*u_i + 1)
                                                *(w*w_i + 1)
                                                *(u*u_i
                                                + 2*v*v_i
                                                + w*w_i - 1.0))

                ret_vals[:, i, 2] = 0.125*(w_i*(u*u_i + 1)
                                                *(v*v_i + 1)
                                                *(u*u_i
                                                + v*v_i
                                                + 2*w*w_i - 1.0))

            # Hex32 Element
            elif num_nodes == 32:
                ret_vals[:, i, 0] = (0.28125*u*(u*u_i + 1)
                                    + 0.015625*u_i*(9*u**2
                                                    + 9*v**2
                                                    + 9*w**2 - 19.0))*(v*v_i + 1)*(w*w_i + 1)

                ret_vals[:, i, 1] = (u*u_i + 1)*(0.28125*v*(v*v_i + 1)
                                                    + 0.015625*v_i*(9*u**2
                                                                    + 9*v**2
                                                                    + 9*w**2 - 19.0))*(w*w_i + 1)

                ret_vals[:, i, 2] = (u*u_i + 1)*(v*v_i + 1)*(0.28125*w*(w*w_i + 1)
                                                                    + 0.015625*w_i*(9*u**2
                                                                                    + 9*v**2
                                                                                    + 9*w**2 - 19.0))


            else:
                print('Hexahedral element with {} not found!'.format(num_nodes))

        # corner edge || w
        if types[i] == 2:

            # Hex20 Element
            if num_nodes == 20:
                ret_vals[:, i, 0] = -0.25*u_i*(w**2 - 1)*(v*v_i + 1)
                ret_vals[:, i, 1] = -0.25*v_i*(w**2 - 1)*(u*u_i + 1)
                ret_vals[:, i, 2] = -0.5*w*(u*u_i + 1)*(v*v_i + 1)

            # Hex32 Element
            elif num_nodes == 32:
                ret_vals[:, i, 0] = -0.140625*u_i*(w**2 - 1)*(v*v_i + 1)*(9*w*w_i + 1)
                ret_vals[:, i, 1] = -0.140625*v_i*(w**2 - 1)*(u*u_i + 1)*(9*w*w_i + 1)
                ret_vals[:, i, 2] = -(u*u_i + 1)*(v*v_i + 1)*(0.28125*w*(9*w*w_i + 1) + 1.265625*w_i*(w**2 - 1))

            else:
                print('Hexahedral element with {} not found!'.format(num_nodes))


        # corner edge || u
        if types[i] == 3:

            # Hex20 Element
            if num_nodes == 20:
                ret_vals[:, i, 0] = -0.5*u*(v*v_i + 1)*(w*w_i + 1)
                ret_vals[:, i, 1] = -0.25*v_i*(u**2 - 1)*(w*w_i + 1)
                ret_vals[:, i, 2] = -0.25*w_i*(u**2 - 1)*(v*v_i + 1)

            # Hex32 Element
            elif num_nodes == 32:
                ret_vals[:, i, 0] = -(0.28125*u*(9*u*u_i + 1) + 1.265625*u_i*(u**2 - 1))*(v*v_i + 1)*(w*w_i + 1)
                ret_vals[:, i, 1] = -0.140625*v_i*(u**2 - 1)*(9*u*u_i + 1)*(w*w_i + 1)
                ret_vals[:, i, 2] = -0.140625*w_i*(u**2 - 1)*(9*u*u_i + 1)*(v*v_i + 1)

            else:
                print('Hexahedral element with {} not found!'.format(num_nodes))

        # corner edge || v
        if types[i] == 4:

            # Hex20 Element
            if num_nodes == 20:
                ret_vals[:, i, 0] = -0.25*u_i*(v**2 - 1)*(w*w_i + 1)
                ret_vals[:, i, 1] = -0.5*v*(u*u_i + 1)*(w*w_i + 1)
                ret_vals[:, i, 2] = -0.25*w_i*(v**2 - 1)*(u*u_i + 1)

            # Hex32 Element
            elif num_nodes == 32:
                ret_vals[:, i, 0] = -0.140625*u_i*(v**2 - 1)*(9*v*v_i + 1)*(w*w_i + 1)
                ret_vals[:, i, 1] = -(u*u_i + 1)*(0.28125*v*(9*v*v_i + 1) + 1.265625*v_i*(v**2 - 1))*(w*w_i + 1)
                ret_vals[:, i, 2] = -0.140625*w_i*(v**2 - 1)*(u*u_i + 1)*(9*v*v_i + 1)

            else:
                print('Hexahedral element with {} not found!'.format(num_nodes))


    return ret_vals

def evaluate_hexahedron(element, nodes, phi):
    '''Evaluate a hexahedral element.

    :param element:
        The connectivity of the element.

    :param nodes:
        The (all) nodal coordinates.

    :param phi:
        The shape functions of the finite element.

    :return:
        The points in an (M x 3) numpy array.
    '''
    # this is the number of nodes for the finite element
    num_nodes = len(element)

    # this is the number of evaluation points
    num_eval = phi.shape[0]

    # this is the return data
    p = np.zeros((num_eval, 3))

    # fill it
    for i in range(num_nodes):
        for j in range(3):
            p[:, j] += nodes[element[i], j]*phi[:, i]

    return p

def compute_J(element, nodes, d_phi):
    '''Compute the Jacobian matrix,
    given the connectivity of a finite element and the nodal coordinates,
    as well as the derivatives of the element basis functions.

    :param element:
        The connectivity of the element.

    :param nodes:
        The (all) nodal coordinates.

    :param d_phi:
        The derivatives of the shape functions of the finite element.

    :return:
        The Jacobian in an (M x 3 x 3) numpy array.
    '''

    # this is the number of nodes for the finite element
    num_nodes = len(element)

    # this is the number of evaluation points
    num_eval = d_phi.shape[0]

    # this is the return data
    J = np.zeros((num_eval, 3, 3))

    # fill it
    for i in range(num_nodes):
        for j in range(3):
            for k in range(3):
                J[:, j, k] += nodes[element[i], j]*d_phi[:, i, k]

    return J

def compute_J_inv(J):
    '''Invert a numpy array of Jacobian matrices.

    :param J:
        A numpy array of Jacobian matrices (M x 3 x 3).

    :return:
        The inverse of these matrices.
    '''

    # the number of evaluation points
    num_eval = J.shape[0]

    # the return data
    inv_J = 0.0*J

    for i in range(num_eval):
        inv_J[i, :, :] = np.linalg.inv(J[i, :, :])

    return inv_J

def compute_J_det(J):
    '''Compute the determinants of an array of Jacobian matrices. 

    :param J:
        A numpy array of Jacobian matrices (M x 3 x 3).

    :return:
        The inverse of these matrices.
    '''

    # the number of evaluation points
    num_eval = J.shape[0]

    # the return data
    det_J = np.zeros((num_eval, ))

    for i in range(num_eval):
        det_J[i] = np.linalg.det(J[i, :, :])

    return det_J


def assemble_curl(d_phi):
    '''Assemble the curls of the vector valued basis
    functions

        Phi_ij =  phi_i * e_j  

    for all scalar basis functions phi_i and all 
    three dimensions j = u, v, w.

    Notice that:

                    (      0      )
    curl Phi_iu =   (  d_dw phi_i )
                    ( -d_dv phi_i )
    
                    ( -d_dw phi_i )
    curl Phi_iv =   (      0      )
                    (  d_du phi_i )

                    (  d_dv phi_i )
    curl Phi_iw =   ( -d_du phi_i )
                    (      0      )
    
    :param d_phi:
        The derivatives obtained from the function
        eval_gradient_hexahedron.
    
    :return:
        An (M x 3*num_nodes x 3) numpy array of curl evaluations
    '''

    # the number of evaluations
    num_eval = d_phi.shape[0]

    # the number of nodes
    num_nodes = d_phi.shape[1]

    # make space for the return data
    curls = np.zeros((num_eval, 3*num_nodes, 3))

    # a running index
    k = 0

    # loop over the nodes
    for i in range(num_nodes):

        # u-component
        curls[:, k, 1] = d_phi[:, i, 2]
        curls[:, k, 2] = -d_phi[:, i, 1]

        # v-component
        curls[:, k+1, 0] = -d_phi[:, i, 2]
        curls[:, k+1, 2] = d_phi[:, i, 0]

        # w-component
        curls[:, k+2, 0] = d_phi[:, i, 1]
        curls[:, k+2, 1] = -d_phi[:, i, 0]

        k += 3

    return curls

def assemble_grad(d_phi):
    '''Assemble a matrix with the gradients for the vector potential
    Ansatz.

    :param d_phi:
        The derivatives from eval_gradient_hexahedron.

    :return:
        An (M x 3*num_nodes x 3) numpy array of grad evaluations
    '''

    # the number of evaluations
    num_eval = d_phi.shape[0]

    # the number of nodes
    num_nodes = d_phi.shape[1]

    # make space for the return data
    grad = np.zeros((num_eval, 3*num_nodes, 3))

    # filling this array is simple
    grad[:, ::3, :] = d_phi
    grad[:, 1::3, :] = d_phi
    grad[:, 2::3, :] = d_phi

    return grad

def assemble_divergence(d_phi):
    '''Assemble the divergences of the vector valued basis
    functions

        Phi_ij =  phi_i * e_j  

    for all scalar basis functions phi_i and all 
    three dimensions j = u, v, w.

    Notice that:

        div Phi_ij = d_dj phi_i

    for j = u, v, w.
    
    :param d_phi:
        The derivatives obtained from the function
        eval_gradient_hexahedron.
    
    :return:
        An (M x 3*num_nodes x 1) numpy array of curl evaluations
    '''

    # the number of evaluations
    num_eval = d_phi.shape[0]

    # the number of nodes
    num_nodes = d_phi.shape[1]

    # make space for the return data
    divs = np.zeros((num_eval, 3*num_nodes, 1))

    # a running index
    k = 0

    # loop over the nodes
    for i in range(num_nodes):

        # u-component
        divs[:, k, 0] = d_phi[:, i, 0]

        # v-component
        divs[:, k+1, 0] = d_phi[:, i, 1]

        # w-component
        divs[:, k+2, 0] = d_phi[:, i, 2]

        k += 3

    return divs


def eval_shape_hexahedron_compl(points, order):
    '''Evaluate the Lagrange shape functions in a Hexahedron. This
    function is for the complete basis and uses gmsh.
    We dont use it currently, but it could become handy in the future.
    
    :param points:
        A numpy array of dimension (M, 3) with the local coordinates (u,v,w)
        in (-1, 1)^3 in each row.

    :param num_nodes:
        The order of the finite element space.

    :return:
        An (M x num_nodes) numpy array of shape function evaluations.
    '''

    # the number of evaluation positions
    M = points.shape[0]

    # initialize gmsh if not done already
    if not gmsh.isInitialized():
            gmsh.initialize()

    _, phi, _ = gmsh.model.mesh.getBasisFunctions(5, points.flatten(), 'Lagrange' + str(order))

    # the number of basis functions
    K = np.int32(len(phi)/M)

    phi.shape = (M, K)

    return phi

def eval_shape_hexahedron_compl(points, order):
    '''Evaluate the gradient of the shape functions of a Hexahedral finite
    element. This function is for the complete basis and uses gmsh.
    We dont use it currently, but it could become handy in the future.

    :param points:
        A numpy array of dimension (M, 3) with the local coordinates (u,v,w)
        in (-1, 1)^3 in each row.

    :param num_nodes:
        The order of the finite element space.

    :return:
        An (M x num_dofs x 3) numpy array of shape function gradient
        evaluations.
    '''
    # the number of evaluation positions
    M = points.shape[0]

    # initialize gmsh if not done already
    if not gmsh.isInitialized():
            gmsh.initialize()

    _, grad_phi, _ = gmsh.model.mesh.getBasisFunctions(5, points.flatten(), 'GradLagrange' + str(order))


    # the number of basis functions
    K = np.int32(len(grad_phi)/M/3)

    grad_phi.shape = (M, K, 3)

    return grad_phi

def heal_element(nodes, element, threshold=1e-7, mesh_format='hmo'):
    '''Center the edge nodes of a finite element.
    So far only 20 noded bricks are implemented.

    :param nodes:
        The nodal coordinates. 

    :param element:
        The connectivity of the finite element.

    :param threshold:
        A threshold for the allowed distance between the
        edge mid point and the center between the two corner nodes.

    :param format:
        The mesh format, default hmo.

    :return:
        The healed nodal coordinates.
    '''

    # make a local copy of the element
    el = element.copy()

    if mesh_format == 'hmo':
        el -= 1
    else:
        print('Warning! Unknown mesh format {}!'.format(mesh_format))

    # copy the original node array
    nodes_c = nodes.copy()

    # get the number of nodes
    num_nodes = len(el)

    if num_nodes == 20:
        # in this array, the rows represent the edges of the
        # finite element. The first and last columns specify the
        # corner nodes of the edge, the central column specifies the
        # mid node.
        edges = np.array([[0, 1, 2],
                        [2, 3, 4],
                        [4, 5, 6],
                        [6, 7, 0],
                        [0, 8, 12],
                        [2, 9, 14],
                        [4, 10, 16],
                        [6, 11, 18],
                        [12, 13, 14],
                        [14, 15, 16],
                        [18, 17, 16],
                        [12, 19, 18]], dtype=np.int64)
                
    else:
        print('No routine for plotting elements with {} nodes is implemented yet!'.format(num_nodes))


    for ee in edges:
        # the center between the corner nodes
        center = 0.5*(nodes[el[ee[0]], :] + nodes[el[ee[2]], :])
        # the distance of the mid node to this one
        dist = np.linalg.norm(center-nodes[el[ee[1]], :])
        
        if dist > threshold:

            nodes_c[el[ee[1]], :] = center

    return nodes_c


    

def plot_element(ax, nodes, element, linewidth=1.0, markersize=2.0, linestyle='-o', color='k', mesh_format='hmo'):
    '''Plot the finite element into an existing matplotlib axis.
    
    :param ax:
        The axes to plot into.
         
    :param nodes:
        All nodal coordinates of the mesh.

    :param element:
        The finite element connectivity. So far only 20 noded bricks are allowed.

    :param linewidth:
        The linewidth.

    :param markersize:
        The markersize for to highlight the nodes.

    :param linestyle:
        A string to specify the linestyle. Default '-o' i.e. a solid line with nodes.

    :param color:
        The color to plot the element.

    :param mesh_format:
        The mesh format. Default is hmo, where the first node has index 1.
    
    :return:
        Nothing.
    '''

    # make a local copy of the element
    el = element.copy()

    if mesh_format == 'hmo':
        el -= 1
    else:
        print('Warning! Unknown mesh format {}!'.format(mesh_format))

    # get the number of nodes
    num_nodes = len(el)

    if num_nodes == 20:
        edges = np.array([[0, 1],
                        [1, 2],
                        [2, 3],
                        [3, 4],
                        [4, 5],
                        [5, 6],
                        [6, 7],
                        [7, 0],
                        [0, 8],
                        [8, 12],
                        [2, 9],
                        [9, 14],
                        [4, 10],
                        [10, 16],
                        [6, 11],
                        [11, 18],
                        [12, 13],
                        [13, 14],
                        [14, 15],
                        [15, 16],
                        [18, 17],
                        [17, 16],
                        [12, 19],
                        [19, 18]], dtype=np.int64)
        
    else:
        print('No routine for plotting elements with {} nodes is implemented yet!'.format(num_nodes))

        return
    
    for ee in edges:
        ax.plot([nodes[el[ee[0]], 0], nodes[el[ee[1]], 0]],
                [nodes[el[ee[0]], 1], nodes[el[ee[1]], 1]],
                [nodes[el[ee[0]], 2], nodes[el[ee[1]], 2]], linestyle,
                markersize=markersize, linewidth=linewidth, color=color)
        
    return 

if __name__ == "__main__":

    # ===============================
    # This is a test to debug the
    # 32 noded brick elements
    # ===============================
    
    nodes = hex_nodal_coordinates(32)

    types = hex_node_types(32)

    element = np.linspace(0, 31, 32, dtype=np.int32)

    def u_gt(x, y, z):
        return y

   
    phi =  np.zeros((32, ))

    # make a meshgrid
    resol = 10

    X, Y, Z = np.meshgrid(np.linspace(-1, 1, resol),
                          np.linspace(-1, 1, resol),
                          np.linspace(-1, 1, resol))
    
    points = np.zeros((resol**3, 3))
    points[:, 0] = X.flatten()
    points[:, 1] = Y.flatten()
    points[:, 2] = Z.flatten()
    
    u_ana = u_gt(points[:, 0], points[:, 1], points[:, 2])

    U_eval = eval_shape_hexahedron(points, nodes, types)

    dU_eval = eval_gradient_hexahedron(points, nodes, types)

    u_eval = 0.0*u_ana
    du_eval = np.zeros((resol**3, 3))

    for i in range(32):
        this_u = u_gt(nodes[i, 0], nodes[i, 1], nodes[i, 2])
        u_eval += this_u*U_eval[:, i]

        du_eval[:, 0] += this_u*dU_eval[:, i, 0]
        du_eval[:, 1] += this_u*dU_eval[:, i, 1]
        du_eval[:, 2] += this_u*dU_eval[:, i, 2]
    
    
    print(np.linalg.norm(u_ana-u_eval))

    print(du_eval)