import numpy as np
import gmsh

class quad_Q8():
    '''This class provides all functionality for 8 noded quad elements.
    '''

    def __init__(self):
        '''Default constructor. It initialized gmsh if this was not done
        already.
        '''
        if not gmsh.isInitialized():
            gmsh.initialize()

        # this is the gmsh code for this type of element
        self.gmsh_code = 16

    def evaluate_basis(self, q):
        '''Evaluate the basis functions.
        
        :param q:
            The local coordinates in an array with M*3 elements, where
            M is the number of points. This is the default gmsh style, where
            all quadrature points are always given in u, v, w, i.e. 3D local
            coordinates.

        :return:
            The eight basis functions evaluated at the M points.
            i.e. an (M x 8) array.
        '''
        
        # this is the number of evaluation points
        M = np.int32(len(q)/3)

        # evaluate the basis functions
        _, basis, _ = gmsh.model.mesh.getBasisFunctions(self.gmsh_code, q, "Lagrange")

        # reshape
        basis.shape = (M, 8)

        # return
        return basis
        
    def evaluate_basis_derivative(self, q):
        '''Evaluate the derivatives of the basis functions.
        
        :param q:
            The local coordinates in an array with M*3 elements, where
            M is the number of points. This is the default gmsh style, where
            all quadrature points are always given in u, v, w, i.e. 3D local
            coordinates.

        :return:
            The eight basis function derivatives evaluated at the M points.
            i.e. an (M x 8 x 2) array.
        '''

        # this is the number of evaluation points
        M = np.int32(len(q)/3)

        # evaluate the basis functions
        _, basis, _ = gmsh.model.mesh.getBasisFunctions(self.gmsh_code, q, "GradLagrange")

        # reshape
        basis.shape = (M, 8, 3)

        # return
        return basis

    def evaluate(self, c, nodes, basis):
        '''Evaluate the positions given the evaluated basis
        functions at some local coordinates.
        
        :param c:
            The node connectivity of this element.
        
        :param nodes:
            The nodes.

        :param basis:
            The basis functions evaluated at the local coordinates.

        :return:
            The positions at the M evaluation points in an M x 3 array.
        '''

        # the number of evaluation points
        num_eval = basis.shape[0]

        # return values
        ret_val = np.zeros((num_eval, 3))
        
        # add basis functions
        for i in range(8):
            ret_val[:, 0] += basis[:, i]*nodes[c[i], 0]
            ret_val[:, 1] += basis[:, i]*nodes[c[i], 1]
            ret_val[:, 2] += basis[:, i]*nodes[c[i], 2]

        return ret_val
    
    def evaluate_derivative(self, c, nodes, basis_der):
        '''Evaluate the spatial derivative given the evaluated basis
        functions at some local coordinates.
        
        :param c:
            The node connectivity of this element.
        
        :param nodes:
            The nodes.

        :param basis:
            The basis functions evaluated at the local coordinates.

        :return:
            The positions at the M evaluation points in an M x 3 array.
        '''

        # the number of evaluation points
        num_eval = basis_der.shape[0]

        # return values
        ret_val = np.zeros((num_eval, 3, 2))

        # add basis functions
        for i in range(8):
            ret_val[:, 0, 0] += basis_der[:, i, 0]*nodes[c[i], 0]
            ret_val[:, 1, 0] += basis_der[:, i, 0]*nodes[c[i], 1]
            ret_val[:, 2, 0] += basis_der[:, i, 0]*nodes[c[i], 2]

            ret_val[:, 0, 1] += basis_der[:, i, 1]*nodes[c[i], 0]
            ret_val[:, 1, 1] += basis_der[:, i, 1]*nodes[c[i], 1]
            ret_val[:, 2, 1] += basis_der[:, i, 1]*nodes[c[i], 2]

        return ret_val
    
    def interpolate(self, c, vals, basis):
        '''Interpolate a function given at the nodes of the mesh.

        :param c:
            The finite element connectivity.

        :param vals:
            The valued to interpolate.
        
        :param basis:
            The basis functions already evaluated at the quadrature points.

        :return:
            The interpolated function.
        '''

        # the dimension of the function
        dim = vals.shape[1]

        # the number of evaluation points
        num_eval = basis.shape[0]

        # return values
        ret_val = np.zeros((num_eval, dim))

        # add basis functions
        for i in range(8):
            for j in range(dim):
                ret_val[:, j] += basis[:, i]*vals[c[i], j]

        return ret_val

    def evaluate_surface_element(self, c, nodes, basis_der):
        '''Evaluate the surface elementsgiven the evaluated basis
        functions at some local coordinates.
        
        :param c:
            The node connectivity of this element.
        
        :param nodes:
            The nodes.

        :param basis:
            The basis functions evaluated at the M local coordinates.

        :return:
            The surface elements in an array with M elements, and the 
            normal vecotrs in an M x 3 array.
        '''

        # get the derivatives
        der = self.evaluate_derivative(c, nodes, basis_der)

        # compute the cross products
        cross = np.empty_like(der[:, :, 0])

        cross[:, 0] = der[:, 1, 0]*der[:, 2, 1] - der[:, 2, 0]*der[:, 1, 1]
        cross[:, 1] = der[:, 2, 0]*der[:, 0, 1] - der[:, 0, 0]*der[:, 2, 1]
        cross[:, 2] = der[:, 0, 0]*der[:, 1, 1] - der[:, 1, 0]*der[:, 0, 1]

        # the surface element
        surf_el = np.linalg.norm(cross, axis=1)

        # the normal vector
        cross[:, 0] /= surf_el
        cross[:, 1] /= surf_el
        cross[:, 2] /= surf_el

        return surf_el, cross
    
    def get_quadrarure_rule(self, order):
        '''Get the quadrature nodes and weigths for
        this element.
        
        :param order:
            The order of the quadrature rule.
            
        :retrurn:
            The array of M weights and the local coordinates of the points in an M x 2 array.
        '''

        q, weights = gmsh.model.mesh.getIntegrationPoints(self.gmsh_code, "Gauss" + str(order))

        return weights, q