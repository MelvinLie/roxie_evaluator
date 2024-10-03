import numpy as np
import os
import matplotlib.pyplot as plt
import pyvista as pv
import gmsh
from tqdm import tqdm
import numpy.matlib as matlib
import scipy.interpolate as interpolate 

from .boundary_element import boundary_element
from .cpp_mod import compute_B_eddy_ring, compute_B_eddy_ring_mat

# permeability of free space
mu_0 = 4.0*np.pi*1e-7

class EddyRing():

    def __init__(self, mesh):
        '''Default constructor.
        
        :param mesh:
            A gmsh mesh object.

        :return:
            None.
        '''

        _, nodes, _ = mesh.getNodes()
        self.nodes = np.array(nodes)
        self.nodes.shape = (np.int64(len(nodes)/3), 3)

        self.cell_types, _, self.cells = gmsh.model.mesh.getElements(2)

        # the number of nodes
        num_nodes = self.nodes.shape[0]

        print('num_nodes = {}'.format(num_nodes))


        return None
    
    def setup_mesh(self, nodes, cells, cell_types):
        """Setup the mesh from external information.

        :param nodes:
            The nodal coordinates.

        :param cells:
            The mesh connectivity.

        :param cell_types.
            The mesh cell types.

        :return:
            None.
        """

        self.nodes = nodes
        self.cells = cells
        self.cell_types =cell_types

        return None

    
    def get_num_dofs(self):
        '''Get the total number of degrees of freedom.
        
        :return:
            An integer specifying the number if DoFs.
        '''
        return self.nodes.shape[0]

    def make_all_interpolations(self, x, quad_order=8):
        '''Make all the interpolations for the evaluation of the eddy ring potential.

        :param x:
            The density vector.
        
        :param quad_order:
            The order of the quadrature rule.
            
        :return:
            The quadrature points (q), the normal vectors (n), the integration weights (w).
            If u was given we return also the interpolated potentials.
            If dn_u was given we return also the interpolated normal derivatives.
        '''

        # allocate the interpolations for all cell types

        # make source points
        q_all = np.zeros((0, 3))
        # make source vectors
        s_all = np.zeros((0, 3))
        # make source weights
        w_all = np.zeros((0, ))

        # loop over cell types
        for i, ct in enumerate(self.cell_types):

            # make the boundary element
            b_element = boundary_element(ct)

            # get the number of nodes for this boundary element
            num_nodes = b_element.get_number_of_nodes()

            # determine the number of elements of this type
            num_el = np.int64(len(self.cells[i])/num_nodes) 

            # get the quadrature rules
            w, q = b_element.get_quadrarure_rule(quad_order)

            # the number of quadrature points
            num_quad = np.int32(len(q)/3)

            # evaluate the basis functions
            N = b_element.evaluate_basis(q)

            # and also the derivatives
            dN = b_element.evaluate_basis_derivative(q)

            # allocate the interpolations for this cell type

            # make source points
            q_i = np.zeros((num_el*num_quad, 3))
            # make source vectors
            s_i = np.zeros((num_el*num_quad, 3))
            # make source weights
            w_i = np.zeros((num_el*num_quad, ))

            # this index points to the first node of the current element
            cell_idx = 0

            # this is an index for the souces
            src_idx = 0

            # loop over all elements of this type
            for j in range(num_el):

                # notice that gmsh starts counting at 1
                q_i[src_idx:src_idx+num_quad, :] = b_element.evaluate(self.cells[i][cell_idx:cell_idx+num_nodes]-1, self.nodes, N)
                s_i[src_idx:src_idx+num_quad, :] = b_element.evaluate_surface_curl(x, self.cells[i][cell_idx:cell_idx+num_nodes]-1, self.nodes, dN)
                
                w_i[src_idx:src_idx+num_quad] = w
                   
                src_idx += num_quad
                cell_idx += num_nodes
                
            # append the sources to the arrays
            q_all = np.append(q_all, q_i, axis=0)
            s_all = np.append(s_all, s_i, axis=0)
            w_all = np.append(w_all, w_i, axis=0)

        return q_all, s_all, w_all
    

    def compute_normal_vectors(self, quad_order=1):
        '''Evaluate the normal vectors on the boundary elements.
        
        :param quad_order:
            The quadrature order.

        :return:
            The evaluation points and all normal vectors.
        '''
        # allocate the interpolations for all cell types

        # make source points
        n_all = np.zeros((0, 3))
        # make source points
        q_all = np.zeros((0, 3))

        # loop over cell types
        for i, ct in enumerate(self.cell_types):

            # make the boundary element
            b_element = boundary_element(ct)

            # get the number of nodes for this boundary element
            num_nodes = b_element.get_number_of_nodes()

            # determine the number of elements of this type
            num_el = np.int64(len(self.cells[i])/num_nodes) 

            # get the quadrature rules
            w, q = b_element.get_quadrarure_rule(quad_order)

            # the number of quadrature points
            num_quad = np.int32(len(q)/3)

            # evaluate the basis functions
            N = b_element.evaluate_basis(q)
            
            # and also the derivatives
            dN = b_element.evaluate_basis_derivative(q)

            # allocate the interpolations for this cell type

            # make source points
            n_i = np.zeros((num_el*num_quad, 3))
            q_i = np.zeros((num_el*num_quad, 3))

            # this index points to the first node of the current element
            cell_idx = 0

            # this is an index for the souces
            src_idx = 0

            # loop over all elements of this type
            for j in range(num_el):

                # notice that gmsh starts counting at 1
                q_i[src_idx:src_idx+num_quad, :] = b_element.evaluate(self.cells[i][cell_idx:cell_idx+num_nodes]-1, self.nodes, N)
                _, n_i[src_idx:src_idx+num_quad, :] =  b_element.evaluate_surface_element(self.cells[i][cell_idx:cell_idx+num_nodes]-1, self.nodes, dN)
                                   
                src_idx += num_quad
                cell_idx += num_nodes
                
            # append the sources to the arrays
            q_all = np.append(q_all, q_i, axis=0)
            n_all = np.append(n_all, n_i, axis=0)

        return q_all, n_all

    def compute_B(self, points, x, quad_order=8):
        '''Compute the flux density at given points in 3D.

        :param points:
            The evaluation points.

        :param x:
            The density vector.
        
        :param quad_order:
            The order of the quadrature rule.
            
        :return:
            The flux density at the positions points.
        '''

        # make all interpolations first
        q, s, w = self.make_all_interpolations(x, quad_order=quad_order)

        # now compute B
        return compute_B_eddy_ring(points, q, s, w)
    
    def compute_B_mat(self, points, normals, quad_order=8):
        '''Compute the matrix for evaluating the flux density at points in 3D.

        :param points:
            The evaluation points.

        :param normals:
            The normals vectors.
        
        :param quad_order:
            The order of the quadrature rule.
            
        :return:
            The matrix for the flux density evaluation.
        '''

        # make space for the matrix
        H = np.zeros((points.shape[0], self.nodes.shape[0]))

        # loop over cell types
        for i, ct in enumerate(self.cell_types):

            # the cells of this element type
            cells = self.cells[i] - 1

            # make the boundary element
            b_element = boundary_element(ct)

            # get the quadrature rules
            w, q = b_element.get_quadrarure_rule(quad_order)

            # evaluate the basis functions
            N = b_element.evaluate_basis(q)

            # and also the derivatives
            dN = b_element.evaluate_basis_derivative(q)

            # now increment the matrix
            H += compute_B_eddy_ring_mat(points, normals, self.nodes, cells, N, dN, q, w)

        return H