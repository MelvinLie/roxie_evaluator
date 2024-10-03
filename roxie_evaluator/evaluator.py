import numpy as np
import os
import matplotlib.pyplot as plt
import pyvista as pv
import gmsh
from tqdm import tqdm
import numpy.matlib as matlib
import scipy.interpolate as interpolate 
import time

from .read_hmo import read_hmo_file
from .read_hmo import get_number_of_nodes
from .read_hmo import get_vtk_cell_code

from .read_ele import read_ele_file
from .read_ele import get_num_nodes_ele

from .read_solution import read_sol_file
from .read_solution import sort_solution

from .read_cor import read_cor_file

from .quad_element import quad_Q8

from .tri_element import triangle_T6

from .read_cond import read_cond_file

from .cpp_mod import compute_B_cpp
from .cpp_mod import compute_A_and_B_cpp
from .cpp_mod import compute_A_iron_cpp
from .cpp_mod import compute_B_iron_cpp
from .cpp_mod import compute_A_mlfmm
from .cpp_mod import compute_B_mlfmm
from .cpp_mod import compute_B_roxie_mlfmm

from . import hexahedral_elements as hex
from . import integration_tools as int_tools

# permeability of free space
mu_0 = 4.0*np.pi*1e-7

class evaluator():
    '''This is the evaluator class.
    '''

    def __init__(self, directory, filename='bemfem', cond_filename='opera8.roxie', read_solution=True):
        '''Default constructor

        :param directory:
            The directory with the ROXIE results.

        :param filename:
            The filename of this bemfem calculation. Default = bemfem.

        :param read_solution:
            Set this flag if You like to read a solution file.

        :return:
            Nothing.
        '''
        
        filename_hmo  = os.path.join(directory, filename + '.hmo')
        filename_sol  = os.path.join(directory, filename + '.sol')
        filename_ele  = os.path.join(directory, filename + '.ele')
        filename_cor  = os.path.join(directory, filename + '.cor')

        # read the hmo file
        self.comp_names, \
            self.node_numbers, \
                self.nodes, \
                    self.elements, \
                        self.element_codes = read_hmo_file(filename_hmo)
        
        if read_solution:
            # read the solution file
            self.times, \
                self.node_number_list, \
                    self.pot_list, \
                        self.der_list = read_sol_file(filename_sol)
            
            # convert times to numpy array
            self.times = np.array(self.times)

        # read the element file with the boundary element info
        self.c_b, self.comp_b, self.types_b = read_ele_file(filename_ele, self.elements.shape[0])

        # get the table to map node indices from hmo to ele
        self.node_index_table = read_cor_file(filename_cor)

        # read also the conductor bricks
        self.p_cond, self.c_cond, self.curr_cond = read_cond_file(os.path.join(directory, cond_filename))


        # convert everything to m
        self.nodes *= 1e-3
        self.p_cond *= 1e-3

        # setup the symetry flags
        self.sym_xy = 0
        self.sym_xz = 0
        self.sym_yz = 0

        return
    
    def set_symmetry_flags(self, sym_xy, sym_xz, sym_yz):
        '''Set the symmetry flags.

        :param sym_xy:
            Symmetry flag for the xy plane. 0: No symmetry. 1: Bn = 0.
            2: Ht = 0.

        :param sym_xz:
            Symmetry flag for the xz plane. 0: No symmetry. 1: Bn = 0.
            2: Ht = 0.

        :param sym_yz:
            Symmetry flag for the yz plane. 0: No symmetry. 1: Bn = 0.
            2: Ht = 0.

        :return:
            Nothing.
        '''

        self.sym_xy = sym_xy
        self.sym_xz = sym_xz
        self.sym_yz = sym_yz

    # def rotate(self, rot_x, rot_y, rot_z) -> None:
    #     '''Rotate the magnet around the x, y and z axis is this order:
    #         r = R_z*R_y*Rx*r',
    #     where r' is the initial condition and r is the final one.
    #     Dont use this function in combination with symmetry flags!

    #     :param rot_x:
    #         The angle around the x axis for the first rotation in radian.

    #     :param rot_y:
    #         The angle around the y axis for the second rotation in radian.
        
    #     :param rot_z:
    #         The angle around the z axis for the third rotation in radian.

    #     :return:
    #         None.

    #     '''

    #     print('WARNING! Symmetry flag will be violated!')
    #     # make the three rotation matrices
    #     R_x = np.array([[1.0, 0.0, 0.0],
    #                     [0.0, np.cos(rot_x), -np.sin(rot_x)],
    #                     [0.0, np.sin(rot_x), np.cos(rot_x)]])
    #     R_y = np.array([[np.cos(rot_y), 0.0, np.sin(rot_y)],
    #                     [0.0, 1.0, 0.0],
    #                     [-np.sin(rot_y), 0.0, np.cos(rot_y)]])
    #     R_z = np.array([[np.cos(rot_z), -np.sin(rot_z), 0.0],
    #                     [np.sin(rot_z), np.cos(rot_z), 0.0],
    #                     [0.0, 0.0, 1.0]])
          
    #     R = R_z @ R_y @ R_x

    #     # apply rotation
    #     self.p_cond = (R @ self.p_cond.T).T
    #     self.nodes = (R @ self.nodes.T).T

    #     return None

    def get_num_dofs(self):
        '''Get the number of degrees of freedom.
        
        :return:
            The number of degrees of freedom is equal to the number
            of nodes in the mesh.
        '''

        return len(self.pot_list[0])
    
    def get_near_field_distance(self, ratio=0.1):
        '''Get the near field distance based on the conductor geometry.

        :param ratio:
            The ratio. All interactions which are closer than ratio*diam
            will be considered near field, where diam is the domain diameter.

        :return:
            The near field distance.
        '''

        diam = max([max(self.p_cond[:, 0]) - min(self.p_cond[:, 0]),
                    max(self.p_cond[:, 1]) - min(self.p_cond[:, 1]),
                    max(self.p_cond[:, 2]) - min(self.p_cond[:, 2])])
        
        return ratio*diam

    def compute_coil_field(self, points, Nn, Nb, field='A', near_field_ratio=0.5):
        '''Compute the coil field.

        :param points:
            The points to evaluate the coil field.

        :param Nn:
            Number of filaments in N direction.

        :param Nb:
            Number of filaments in B direction.

        :param field:
            A string specifying the field to compute. Options are: 
            "A": magnetic vector potential. "B": Magnetic flux density,
            "all": magnetic vector potential and magnetic flux density.

        :return:
            The field vectors at these positions in an
            (M x 3) numpy array.
        '''

        if field == 'A':

            # so far we dont have a function compute A
            A, B = compute_A_and_B_cpp(self.p_cond,
                                self.c_cond,
                                Nn + 0.0*self.curr_cond, 
                                Nb + 0.0*self.curr_cond,
                                self.curr_cond/Nn/Nb, points, near_field_ratio)
            
            return A
        
        elif field == 'B':

            B = compute_B_cpp(self.p_cond,
                                self.c_cond,
                                Nn + 0.0*self.curr_cond, 
                                Nb + 0.0*self.curr_cond,
                                self.curr_cond/Nn/Nb, points, near_field_ratio)
            
            return B

        elif field == 'all':
            
            A, B = compute_A_and_B_cpp(self.p_cond,
                                self.c_cond,
                                Nn + 0.0*self.curr_cond, 
                                Nb + 0.0*self.curr_cond,
                                self.curr_cond/Nn/Nb, points, 0.0)
            
            return A, B
    
    def compute_iron_field_mlfmm(self, points, field='A', L=6, max_tree_level=2, quad_order=6):
        '''Compute the iron field with the multilevel fast multipole method.

        :param points:
            The points to evaluate the coil field.

        :param L:
            The maximum order of the solid harmonics.

        :param max_tree_level:
            The maximum level of the cluster tree refinement.

        :param quad_order:
            The quadrature order for the Gaussian integration.

        :param field:
            A string specifying the field to compute. Options are: 
            "A": magnetic vector potential. "B": Magnetic flux density,
            "all": magnetic vector potential and magnetic flux density.

        :return:
            The field vectors at these positions in an
            (M x 3) numpy array.
        '''

        # we need to get the solution vectors
        u = sort_solution(self.node_number_list[0], self.pot_list[0])
        dn_u = sort_solution(self.node_number_list[0], self.der_list[0])

        # make all the interpolations
        q, n, w, A, dA = self.make_all_interpolations(u=u, dn_u=dn_u, quad_order=quad_order)
        
        # we need to scale the source vectors with the weights
        for i in range(3):
            A[:, i] *= w
            dA[:, i] *= w

        if field == 'A':
            # compute the field using the fast multipole method
            A_ret = -1e7/4.0/np.pi*compute_A_mlfmm(q, A, points, L, max_tree_lvl=max_tree_level, normals=n, kind='Vector-Valued-Double-Layer')
            A_ret += -1e7/4.0/np.pi*compute_A_mlfmm(q, dA, points, L, max_tree_lvl=max_tree_level, normals=n, kind='Vector-Valued-Single-Layer')

            return A_ret
        
        elif field == 'B':
            # compute the field using the fast multipole method
            B = 1e7/4.0/np.pi*compute_B_mlfmm(q, A, points, L, max_tree_lvl=max_tree_level, normals=n, kind='Vector-Valued-Double-Layer')
            B += -1e7/4.0/np.pi*compute_B_mlfmm(q, dA, points, L, max_tree_lvl=max_tree_level, kind='Vector-Valued-Single-Layer')

            return B
        
        else:
            print('Field {} unknown!'.format(field))
            return None


    def compute_coil_field_mlfmm(self, points, Nn, Nb, field='A', max_seg_len=10e-3, L=6, max_tree_level=2, quad_order=2):
        '''Compute the coil field.

        :param points:
            The points to evaluate the coil field.

        :param Nn:
            Number of filaments in N direction.

        :param Nb:
            Number of filaments in B direction.

        :param L:
            The maximum order of the solid harmonics.

        :param max_tree_level:
            The maximum level of the cluster tree refinement.

        :param quad_order:
            The quadrature order for the Gaussian integration.

        :param field:
            A string specifying the field to compute. Options are: 
            "A": magnetic vector potential. "B": Magnetic flux density,
            "all": magnetic vector potential and magnetic flux density.

        :return:
            The field vectors at these positions in an
            (M x 3) numpy array.
        '''

        # time_start = time.time()

        # make the sources
        source_pts, source_vecs = self.make_all_conductor_sources(Nn, Nb, max_seg_len=max_seg_len, quad_order=quad_order)

        # time_end = time.time()

        # print('elapsed time for conductor refinement = {} sec'.format(time_end - time_start))

        if field == 'A':
            # compute the field using the fast multipole method
            A = compute_A_mlfmm(source_pts, source_vecs, points, L, max_tree_lvl=max_tree_level)

            return A
        
        elif field == 'B':
            # compute the field using the fast multipole method
            B = compute_B_mlfmm(source_pts, source_vecs, points, L, max_tree_lvl=max_tree_level)

            return B
        
        else:
            print('Field {} unknown!'.format(field))
            return None

    def compute_B_bemfem_mlfmm(self, points, Nn, Nb, max_seg_len=10e-3, L=6,
                              max_tree_level=2, quad_order_coil=2, quad_order_iron=6):
        '''Compute the iron and coil B fields with the multilevel fast multipole method.

        :param Nn:
            Number of filaments in N direction.

        :param Nb:
            Number of filaments in B direction.

        :param L:
            The maximum order of the solid harmonics.

        :param max_tree_level:
            The maximum level of the cluster tree refinement.

        :param quad_order_coil:
            The quadrature order for the Gaussian integration for the conductors.

        :param quad_order_iron:
            The quadrature order for the Gaussian integration for the iron.

        :return:
            The field vectors at these positions in an
            (M x 3) numpy array.
        '''

        # make the coil sources
        src_pts_coil, src_vec_coil = self.make_all_conductor_sources(Nn, Nb, max_seg_len=max_seg_len, quad_order=quad_order_coil)

        # we need to get the solution vectors
        u = sort_solution(self.node_number_list[0], self.pot_list[0])
        dn_u = sort_solution(self.node_number_list[0], self.der_list[0])

        # make all the interpolations
        src_pts_iron, normals, w, A, dA = self.make_all_interpolations(u=u, dn_u=dn_u, quad_order=quad_order_iron)
        
        # we need to scale the source vectors with the weights
        for i in range(3):
            A[:, i] *= w
            dA[:, i] *= w

        return compute_B_roxie_mlfmm(src_pts_coil, src_vec_coil, src_pts_iron,
                                        A, dA, normals, points, L,
                                        max_tree_lvl=max_tree_level)

    def make_all_conductor_sources(self, Nn, Nb, max_seg_len=30e-3, quad_order=2):
        '''Make all conductor sources for the field calculation with the MLFMM.

        :param Nn:
            Number of filaments in N direction.

        :param Nb:
            Number of filaments in B direction.

        :param max_seg_len:
            The maximum length of the conductor segments.
            We cut the segments into parts of this size. This
            is because ROXIE may use very long segments in the
            straight section.

        :param quad_order:
            The quadrature order for the Gaussian integration in each line segment.

        :return:
            The source points and the scaled source vectors.
        '''
        
        # we initialize gmsh if not already done
        if not gmsh.isInitialized():
            gmsh.initialize()
            
        # get the quadrature rule
        q, weights = gmsh.model.mesh.get_integration_points(1, "Gauss" + str(quad_order))
        
        # the number of quadrature points
        num_quad = len(weights)

        # reshape the quadrature points (gmsh always gives 3 components)
        q.shape = (num_quad, 3)

        # we refine the conductor mesh so that all cells are of comparable length
        p, c, curr = self.refine_conductor_mesh(max_seg_len=max_seg_len)
        
        # number of bricks
        num_bricks = c.shape[0]

        # allocate the space for the sources
        source_pts = np.zeros((num_bricks*Nn*Nb*num_quad, 3))
        source_vecs = np.zeros((num_bricks*Nn*Nb*num_quad, 3))

        # the transversal local coordinates of the strands
        u = 0.5*(np.linspace(-1.0, 1.0, Nn + 1)[1:] + np.linspace(-1.0, 1.0, Nn + 1)[:-1])
        v = 0.5*(np.linspace(-1.0, 1.0, Nb + 1)[1:] + np.linspace(-1.0, 1.0, Nb + 1)[:-1])

        # make the local integration coordinates
        loc_u, loc_v, loc_w = np.meshgrid(u, v, q[:, 0])
        loc = np.zeros((len(loc_u.flatten()), 3))
        loc[:, 0] = loc_u.flatten()
        loc[:, 1] = loc_v.flatten()
        loc[:, 2] = loc_w.flatten()

        # make also all weigths
        _, _, all_weights = np.meshgrid(u, v, weights)
        all_weights = all_weights.flatten()

        # the local shape functions of hexahedral elements
        phi = hex.eval_shape_hexahedron(loc, 1)
        # we compute also the derivatives
        d_phi = hex.eval_gradient_hexahedron(loc, 1)

        # a source point counter
        src_cnt = 0

        # total number of integration points
        num_pnts_tot = Nn*Nb*num_quad

        # loop over the bricks
        for i, cc in enumerate(c):

            # evaluate the source point
            pnts = hex.evaluate_hexahedron(cc, p, phi)

            # compute the Jacobians
            J = hex.compute_J(cc, p, d_phi)

            # add these points
            source_pts[src_cnt:src_cnt+num_pnts_tot, :] = pnts

            source_vecs[src_cnt:src_cnt+num_pnts_tot, 0] = J[:, 0, 2]*all_weights*curr[i, 0]/Nn/Nb
            source_vecs[src_cnt:src_cnt+num_pnts_tot, 1] = J[:, 1, 2]*all_weights*curr[i, 1]/Nn/Nb
            source_vecs[src_cnt:src_cnt+num_pnts_tot, 2] = J[:, 2, 2]*all_weights*curr[i, 2]/Nn/Nb
            
            src_cnt += num_pnts_tot

            # for j in range(Nn*Nb*num_quad):
            #     source_pts[src_cnt, :] = pnts[j, :]
            #     source_vecs[src_cnt, :] = J[j, :, 2]*all_weights[j]*curr[i]

            #     src_cnt += 1

        return source_pts, source_vecs

    def refine_conductor_mesh(self, max_seg_len=30e-3, overwrite=False):
        '''Refine the conductor mesh so that all bricks are shorter than max_seg_len.
        Notice that this function returns a conductor mesh with node multiplicity.
        We want to do this, because for us it does not matter and we achieve a clean code.

        :param max_seg_len:
            The maximum length for the segments.

        :param overwrite:
            Set this flag to true if You like to overwrite the existing conductor mesh.

        :return:
            The new mesh points and connectivity and also the conductor currents.
        '''
        
        # we allocate enough space so that ever element can be refined 100 times
        num_cells = self.c_cond.shape[0]
        num_nodes_alloc = num_cells*8*100
        num_cells_alloc = num_cells*100

        # an array with new nodes
        p_new = np.zeros((num_nodes_alloc, 3))

        # an array for the conductor currents
        curr_new = np.zeros((num_cells_alloc, 3))

        # number of bricks
        num_bricks = self.c_cond.shape[0]

        # make a copy of the original mesh
        p_c = self.p_cond.copy()
        c_c = self.c_cond.copy()

        # this is a counter for the nodes
        node_cnt = 0

        # this is a counter for the cells
        cell_cnt = 0

        # loop over the bricks
        for i, c in enumerate(self.c_cond):

            # compute the difference vectors on the sides of the elements
            d_1 = self.p_cond[c[4], :] - self.p_cond[c[0], :]
            d_2 = self.p_cond[c[5], :] - self.p_cond[c[1], :]
            d_3 = self.p_cond[c[6], :] - self.p_cond[c[2], :]
            d_4 = self.p_cond[c[7], :] - self.p_cond[c[3], :]

            # compute the four sidelengths
            len_1 = np.linalg.norm(d_1)
            len_2 = np.linalg.norm(d_2)
            len_3 = np.linalg.norm(d_3)
            len_4 = np.linalg.norm(d_4)


            # the maximum length
            max_len = max([len_1, len_2, len_3, len_4])

            # the numeber of cuts required
            num_cuts = 0

            # check if it exceeds the limits
            if max_len > max_seg_len:

                # determine the number of cuts neccessary
                num_cuts = np.int32(max_len/max_seg_len)

            # this is where the new nodes will be placed between the old ones
            w = np.linspace(0., 1., num_cuts+2)

            for j in range(0, num_cuts+1):
                p_new[node_cnt,     :] = self.p_cond[c[0], :] + w[j]*d_1
                p_new[node_cnt + 1, :] = self.p_cond[c[1], :] + w[j]*d_2
                p_new[node_cnt + 2, :] = self.p_cond[c[2], :] + w[j]*d_3
                p_new[node_cnt + 3, :] = self.p_cond[c[3], :] + w[j]*d_4
                p_new[node_cnt + 4, :] = self.p_cond[c[0], :] + w[j+1]*d_1
                p_new[node_cnt + 5, :] = self.p_cond[c[1], :] + w[j+1]*d_2
                p_new[node_cnt + 6, :] = self.p_cond[c[2], :] + w[j+1]*d_3
                p_new[node_cnt + 7, :] = self.p_cond[c[3], :] + w[j+1]*d_4

                curr_new[cell_cnt] = self.curr_cond[i]
                
                node_cnt += 8
                cell_cnt += 1

                if (node_cnt > num_nodes_alloc):
                    print('Error! The conductor mesh is refined too much! Memory overflow!')


        # filter out the remaining zeros
        p_new = p_new[:node_cnt, :]
        curr_new = curr_new[:cell_cnt, :]


        # now make also a new connectivity
        c_new = np.zeros((cell_cnt, 8), dtype=np.int32)

        # a stencil
        stencil = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int32)

        for i in range(cell_cnt):
            c_new[i, :] = stencil + 8*i

        if overwrite:
            self.p_cond = p_new
            self.c_cond = c_new
            self.curr_cond = curr_new

        return p_new, c_new, curr_new


    def make_all_interpolations(self,
                                    u=np.zeros((0, )),
                                    dn_u=np.zeros((0, )),
                                    quad_order=8):
        '''Make all the interpolations for the evaluation of the single and double
        layer potentials.

        :param u:
            The nodal values of the magnetic vector potential. Default np.zeros((0, )).
            A zero size array means that the potential is not interpolated.

        :param dn_u:
            The nodal values of the derivative of the magnetic vector potential.
            Default np.zeros((0, )). A zero size array means that the derivative is
            not interpolated.

        :param quad_order:
            The order of the quadrature rule.
            
        :return:
            The quadrature points (q), the normal vectors (n), the integration weights (w).
            If u was given we return also the interpolated potentials.
            If dn_u was given we return also the interpolated normal derivatives.
        '''

        interpolate_u = False
        interpolate_dn_u = False

        if len(u) > 0:
            interpolate_u = True

        if len(dn_u) > 0:
            interpolate_dn_u = True

        # get the number of boundary elements
        num_el = self.get_number_of_boundary_elements()

        # get the boundary mesh
        c, vtk_cell_types = self.get_boundary_mesh()

        # make T6 and Q8 elements
        t6 = triangle_T6()
        q8 = quad_Q8()

        # get the quadrature rules
        w_T6, q_T6 = t6.get_quadrarure_rule(quad_order)
        w_Q8, q_Q8 = q8.get_quadrarure_rule(quad_order)

        # evaluate the basis functions
        N_T6 = t6.evaluate_basis(q_T6)
        N_Q8 = q8.evaluate_basis(q_Q8)

        # and also the derivatives
        d_N_T6 = t6.evaluate_basis_derivative(q_T6)
        d_N_Q8 = q8.evaluate_basis_derivative(q_Q8)
        

        # count the number of triangles and quads
        num_T6 = np.sum(vtk_cell_types == 22)
        num_Q8 = np.sum(vtk_cell_types == 23)

        # the number of quadrature points for the two elements
        num_quad_Q8 = np.int32(len(q_Q8)/3)
        num_quad_T6 = np.int32(len(q_T6)/3)

        # counters
        cnt_c = 0
        cnt_q = 0

        # make source points
        q = np.zeros((num_T6*num_quad_T6 + num_Q8*num_quad_Q8, 3))
        n = np.zeros((num_T6*num_quad_T6 + num_Q8*num_quad_Q8, 3))
        w = np.zeros((num_T6*num_quad_T6 + num_Q8*num_quad_Q8, ))

        # the vector potential evaluations and the normal derivaives
        if interpolate_u:
            A = np.zeros((num_T6*num_quad_T6 + num_Q8*num_quad_Q8, 3))
        if interpolate_dn_u:
            dA = np.zeros((num_T6*num_quad_T6 + num_Q8*num_quad_Q8, 3))

        for i in range(num_el):

            # the number of nodes
            num_nodes = c[cnt_c]

            if num_nodes == 6:
                q[cnt_q:cnt_q+num_quad_T6, :] = t6.evaluate(c[cnt_c+1:cnt_c+7], self.nodes, N_T6)
                surf_el, n[cnt_q:cnt_q+num_quad_T6, :] = t6.evaluate_surface_element(c[cnt_c+1:cnt_c+7], self.nodes, d_N_T6)
                w[cnt_q:cnt_q+num_quad_T6] = surf_el*w_T6

                if interpolate_u:
                    A[cnt_q:cnt_q+num_quad_T6, :] = t6.evaluate(c[cnt_c+1:cnt_c+7], u, N_T6)
                if interpolate_dn_u:
                    dA[cnt_q:cnt_q+num_quad_T6, :] = t6.evaluate(c[cnt_c+1:cnt_c+7], dn_u, N_T6)

                cnt_q += num_quad_T6

            elif num_nodes == 8:

                q[cnt_q:cnt_q+num_quad_Q8, :] = q8.evaluate(c[cnt_c+1:cnt_c+9], self.nodes, N_Q8)
                surf_el, n[cnt_q:cnt_q+num_quad_Q8, :] = q8.evaluate_surface_element(c[cnt_c+1:cnt_c+9], self.nodes, d_N_Q8)
                w[cnt_q:cnt_q+num_quad_Q8] = surf_el*w_Q8

                if interpolate_u:
                    A[cnt_q:cnt_q+num_quad_Q8, :] = q8.evaluate(c[cnt_c+1:cnt_c+9], u, N_Q8)
                if interpolate_dn_u:
                    dA[cnt_q:cnt_q+num_quad_Q8, :] = q8.evaluate(c[cnt_c+1:cnt_c+9], dn_u, N_Q8)


                cnt_q += num_quad_Q8

            else:
                print('No element with {} nodes known!'.format(num_nodes))

            cnt_c += num_nodes + 1

        # apply symmetry
        if self.sym_xy > 0:

            q_m = q.copy()
            q_m[:, 2] *= -1.0
            
            n_m = n.copy()
            n_m[:, 2] *= -1.0

            A_m = A.copy()
            dA_m = dA.copy()

            if self.sym_xy == 1:
                A_m[:, 0] *= -1.0
                A_m[:, 1] *= -1.0
                dA_m[:, 0] *= -1.0
                dA_m[:, 1] *= -1.0

            elif self.sym_xy == 2:
                A_m[:, 2] *= -1.0
                dA_m[:, 2] *= -1.0

            else:
                print('symmetry {} unknown'.format(self.sym_xy))

            q = np.append(q, q_m, axis=0)
            n = np.append(n, n_m, axis=0)
            w = np.append(w, w, axis=0)
            if interpolate_u:
                A = np.append(A, A_m, axis=0)
            if interpolate_dn_u:
                dA = np.append(dA, dA_m, axis=0)

        if self.sym_xz > 0:

            q_m = q.copy()
            q_m[:, 1] *= -1.0
            
            n_m = n.copy()
            n_m[:, 1] *= -1.0

            if interpolate_u:
                A_m = A.copy()
            if interpolate_dn_u:
                dA_m = dA.copy()

            if self.sym_xz == 1:
                if interpolate_u:
                    A_m[:, 0] *= -1.0
                    A_m[:, 2] *= -1.0
                if interpolate_dn_u:
                    dA_m[:, 0] *= -1.0
                    dA_m[:, 2] *= -1.0
            elif self.sym_xz == 2:
                if interpolate_u:
                    A_m[:, 1] *= -1.0
                if interpolate_dn_u:
                    dA_m[:, 1] *= -1.0
            else:
                print('symmetry {} unknown'.format(self.sym_xz))

            q = np.append(q, q_m, axis=0)
            n = np.append(n, n_m, axis=0)
            w = np.append(w, w, axis=0)
            if interpolate_u:
                A = np.append(A, A_m, axis=0)
            if interpolate_dn_u:
                dA = np.append(dA, dA_m, axis=0)

        if self.sym_yz > 0:

            q_m = q.copy()
            q_m[:, 0] *= -1.0
            
            n_m = n.copy()
            n_m[:, 0] *= -1.0

            if interpolate_u:
                A_m = A.copy()
            if interpolate_dn_u:
                dA_m = dA.copy()

            if self.sym_yz == 1:
                if interpolate_u:
                    A_m[:, 1] *= -1.0
                    A_m[:, 2] *= -1.0
                if interpolate_dn_u:
                    dA_m[:, 1] *= -1.0
                    dA_m[:, 2] *= -1.0
            elif self.sym_yz == 2:
                if interpolate_u:
                    A_m[:, 0] *= -1.0
                if interpolate_dn_u:
                    dA_m[:, 0] *= -1.0
            else:
                print('symmetry {} unknown'.format(self.sym_yz))

            q = np.append(q, q_m, axis=0)
            n = np.append(n, n_m, axis=0)
            w = np.append(w, w, axis=0)
            if interpolate_u:
                A = np.append(A, A_m, axis=0)
            if interpolate_dn_u:
                dA = np.append(dA, dA_m, axis=0)


        ret_vals = [q, n, w]
        if interpolate_u:
            ret_vals.append(A)
        if interpolate_dn_u:
            ret_vals.append(dA)


        return ret_vals

    def apply_symmetry(self, p, F, field='A'):
        '''Apply the symmetry conditions to a field map.

        :param p:
            The points.

        :param F:
            The field vectors at these points.

        :param field:
            A string which is specifying the field type. Options are
            'A', 'B' or 'n' for magnetic vector potential, magnetic flux
            density or normal vectors.

        :return:
            The updated arrays for points and fields. 
        '''

        if self.sym_xy > 0:

            p_m = p.copy()
            p_m[:, 2] *= -1.0

            F_m = F.copy()
            
            if field == 'n':
                F_m[:, 2] *= -1.0

            elif field == 'A':
                if self.sym_xy == 1:
                    F_m[:, 0] *= -1.0
                    F_m[:, 1] *= -1.0

                elif self.sym_xy == 2:
                    F_m[:, 2] *= -1.0

                else:
                    print('symmetry {} unknown'.format(self.sym_xy))

            elif field == 'B':
                if self.sym_xy == 1:
                    F_m[:, 2] *= -1.0

                elif self.sym_xy == 2:
                    F_m[:, 0] *= -1.0
                    F_m[:, 1] *= -1.0

                else:
                    print('symmetry {} unknown'.format(self.sym_xy))

            p = np.append(p, p_m, axis=0)
            F = np.append(F, F_m, axis=0)

        if self.sym_xz > 0:

            p_m = p.copy()
            p_m[:, 1] *= -1.0
            
            F_m = F.copy()

            if field == 'n':
                F_m[:, 1] *= -1.0

            elif field == 'A':
                if self.sym_xz == 1:
                    F_m[:, 0] *= -1.0
                    F_m[:, 2] *= -1.0
                elif self.sym_xz == 2:
                    F_m[:, 1] *= -1.0
                else:
                    print('symmetry {} unknown'.format(self.sym_xz))

            elif field == 'B':
                if self.sym_xz == 1:
                    F_m[:, 1] *= -1.0
                elif self.sym_xz == 2:
                    F_m[:, 0] *= -1.0
                    F_m[:, 2] *= -1.0
                else:
                    print('symmetry {} unknown'.format(self.sym_xz))

            p = np.append(p, p_m, axis=0)
            F = np.append(F, F_m, axis=0)

        if self.sym_yz > 0:

            p_m = p.copy()
            p_m[:, 0] *= -1.0

            F_m = F.copy()
            
            if field == 'n':
                F_m[:, 0] *= -1.0

            elif field == 'A':
                if self.sym_yz == 1:
                    F_m[:, 1] *= -1.0
                    F_m[:, 2] *= -1.0

                elif self.sym_yz == 2:
                    F_m[:, 0] *= -1.0
                else:
                    print('symmetry {} unknown'.format(self.sym_yz))

            elif field == 'B':
                if self.sym_yz == 1:
                    F_m[:, 0] *= -1.0

                elif self.sym_yz == 2:
                    F_m[:, 1] *= -1.0
                    F_m[:, 2] *= -1.0
                else:
                    print('symmetry {} unknown'.format(self.sym_yz))

            p = np.append(p, p_m, axis=0)
            F = np.append(F, F_m, axis=0)

        return p, F

    def make_boundary_nodes_table(self):
        '''Make a table for the identification of boundary nodes.

        :return:
            The table, an integer numpy array of size (M, 2) where M
            is the number of boundary nodes.
        '''

        # get the number of boundary elements
        num_el = self.get_number_of_boundary_elements()

        # get the boundary mesh
        c, _ = self.get_boundary_mesh()

        # a cell counter
        cnt_c = 0

        # the number of nodes
        num_nodes_mesh = self.nodes.shape[0]

        # visit counter
        visits = np.zeros((num_nodes_mesh, ))

        for i in range(num_el):

            # the number of nodes
            num_nodes = c[cnt_c]


            if num_nodes == 6:
                # count visits
                for j in range(6):
                    visits[c[cnt_c + 1 + j]] += 1.0

                cnt_c += 7

            elif num_nodes == 8:
                # count visits
                for j in range(8):
                    visits[c[cnt_c + 1 + j]] += 1.0

                cnt_c += 9 

        # boundary nodes
        table = np.zeros((num_nodes_mesh, 2), dtype=np.int32)
        table[:, 0] = np.linspace(0, num_nodes_mesh-1, num_nodes_mesh, dtype=np.int32)

        # a boundary node counter
        cnt = 0

        # loop over all nodes
        for i in range(num_nodes_mesh):
            if visits[i] == 0.0:
                table[i, 1] = -1
            else:
                table[i, 1] = cnt
                cnt += 1

        return table

    def compute_mass_matrix(self, quad_order=8):
        '''Compute the mass matrix.

        :param quad_order:
            The order of the numerical quadrature.

        :return:
            The mass matrix.
        '''

        # make the boundary nodes table
        b_node_table = self.make_boundary_nodes_table()

        # count the number of boundary nodes
        num_b_nodes = b_node_table[b_node_table[:, 1] > -1, :].shape[0]

        # make space for the return matrix
        M = np.zeros((num_b_nodes, num_b_nodes))

        # get the number of boundary elements
        num_el = self.get_number_of_boundary_elements()

        # get the boundary mesh
        c, _ = self.get_boundary_mesh()

        # make T6 and Q8 elements
        t6 = triangle_T6()
        q8 = quad_Q8()

        # get the quadrature rules
        w_T6, q_T6 = t6.get_quadrarure_rule(quad_order)
        w_Q8, q_Q8 = q8.get_quadrarure_rule(quad_order)

        # evaluate the basis functions
        N_T6 = t6.evaluate_basis(q_T6)
        N_Q8 = q8.evaluate_basis(q_Q8)
        

        # and also the derivatives
        d_N_T6 = t6.evaluate_basis_derivative(q_T6)
        d_N_Q8 = q8.evaluate_basis_derivative(q_Q8)

        # a counter for the indices in the element list
        cnt_c = 0

        for i in tqdm(range(num_el)):

            # the number of nodes
            num_nodes = c[cnt_c]

            if num_nodes == 6:

                # quadrature points
                q = t6.evaluate(c[cnt_c+1:cnt_c+7], self.nodes, N_T6)

                # surface elements and normal vectors
                surf_el, n = t6.evaluate_surface_element(c[cnt_c+1:cnt_c+7], self.nodes, d_N_T6)

                # loop over all basis functions
                for j in range(6):

                    # the index of this boundary node
                    indx = b_node_table[c[cnt_c+1+j], 1]

                    M[indx, indx] += np.sum(surf_el*w_T6*N_T6[:, j]**2)

            elif num_nodes == 8:

                # quadrature points
                q = q8.evaluate(c[cnt_c+1:cnt_c+9], self.nodes, N_Q8)

                # surface elements and normal vectors
                surf_el, n = q8.evaluate_surface_element(c[cnt_c+1:cnt_c+9], self.nodes, d_N_Q8)

                # loop over all combinations of basis functions
                for j in range(8):
                    # the index of this boundary node
                    indx_j = b_node_table[c[cnt_c+1+j], 1]

                    # loop over all combinations of basis functions
                    for k in range(8):
                        # the index of this boundary node
                        indx_k = b_node_table[c[cnt_c+1+k], 1]

                        M[indx_j, indx_k] += np.sum(surf_el*w_Q8*N_Q8[:, j]*N_Q8[:, k])

            else:
                print('No element with {} nodes known!'.format(num_nodes))

            cnt_c += num_nodes + 1
            
        return M
    
    
    def get_element_average_field(self, solution):
        '''Average the solution on each element.
        
        :param solution:
            The solution vector to average over.

        :return:
            The average values for each element.
        '''

        # We need to sort the cells differently
        c = self.elements.copy()
        c -= 1

        # get the number of boundary elements
        num_el = c.shape[0]

        # the average values
        avgs = np.zeros((num_el, ))

        # loop over all elements
        for i in range(num_el):

            avgs[i] = np.mean(solution[c[i, :]])

        return avgs
    
    def compute_A_in_fem_domain(self, solution=np.array([]), quad_order=1):
        '''Evaluate the magnetic vector potential in the iron
        domain.
        
        :param solution:
            The solution vector.

        :param quad_order:
            The quad_order defines the number of points to evaluate in each finite element.
            Default = 1.
        
        :return:
            Two numpy arrays with three columns. The first array contains the
            evaluation coordinates. The secon column contains the evaluated vector
            potential coordinates at these positions.
        '''

        # check if solution vector is given
        if len(solution) == 0:
            solution, _ = self.get_solution()
            solution = sort_solution(self.node_number_list[0], self.pot_list[0])

        # We need to sort the cells differently
        c = self.elements.copy()
        c -= 1

        # get the number of boundary elements
        num_el = c.shape[0]

        # get the quadrature points
        q, w = int_tools.get_quadrature_rule(quad_order)

        # the number of points per element
        num_pts = q.shape[0]

        # the positions
        points = np.zeros((num_el*num_pts, 3))

        # the vector potentials
        A = np.zeros((num_el*num_pts, 3))

        # evaluate the shape functions at these points (not needed)
        phi = hex.eval_shape_hexahedron(q, 2)

        # the gmsh node definition is different to the one used in hypermesh
        # indx = np.array([0, 8, 1, 12, 5, 16, 4, 10, 9, 11, 18, 17, 3, 13, 2, 14, 6, 19, 7, 15], dtype=np.int64)
        # indx = np.array([0, 2, 14, 12, 6, 4, 16, 18, 1, 8, 7, 9, 3, 13, 15, 19, 5, 11, 10, 17], dtype=np.int64)
        indx = np.array([0, 8, 1, 11, 2, 13, 3, 9, 10, 12, 14, 15, 4, 16, 5, 18, 6, 19, 7, 17], dtype=np.int64)

        # sort
        phi = phi[:, indx]

        # evaluate the derivatives at these points
        # d_phi = hex.eval_gradient_hexahedron(q, 2)

        # loop over all elements
        for i, e in enumerate(c):

            # evaluate this hexahedron for the global position
            points[i*num_pts:(i+1)*num_pts, :] = hex.evaluate_hexahedron(e, self.nodes, phi)

            # evaluate the fem solution
            A[i*num_pts:(i+1)*num_pts, 0] = phi @ solution[e, 0]
            A[i*num_pts:(i+1)*num_pts, 1] = phi @ solution[e, 1]
            A[i*num_pts:(i+1)*num_pts, 2] = phi @ solution[e, 2]

        return points, A
            
    def compute_B_in_fem_domain(self, solution=np.array([]), quad_order=1):
        '''Evaluate the magnetic flux density in the iron
        domain.
        
        :param solution:
            The solution vector.

        :param quad_order:
            The quad_order defines the number of points to evaluate in each finite element.
            Default = 1.
        
        :return:
            Two numpy arrays with three columns. The first array contains the
            evaluation coordinates. The secon column contains the evaluated vector
            potential coordinates at these positions.
        '''

        # check if solution vector is given
        if len(solution) == 0:
            solution, _ = self.get_solution()
            solution = sort_solution(self.node_number_list[0], self.pot_list[0])
            
        # We need to sort the cells differently
        c = self.elements.copy()
        c -= 1

        # get the number of boundary elements
        num_el = c.shape[0]

        # get the quadrature points
        q, w = int_tools.get_quadrature_rule(quad_order)

        # the number of points per element
        num_pts = q.shape[0]

        # the positions
        points = np.zeros((num_el*num_pts, 3))

        # the vector potentials
        B = np.zeros((num_el*num_pts, 3))

        # evaluate the shape functions at these points (not needed)
        phi = hex.eval_shape_hexahedron(q, 2)

        # evaluate the derivatives at these points
        d_phi = hex.eval_gradient_hexahedron(q, 2)
        
        # the gmsh node definition is different to the one used in hypermesh
        # indx = np.array([0, 8, 1, 12, 5, 16, 4, 10, 9, 11, 18, 17, 3, 13, 2, 14, 6, 19, 7, 15], dtype=np.int64)
        indx = np.array([0, 8, 1, 11, 2, 13, 3, 9, 10, 12, 14, 15, 4, 16, 5, 18, 6, 19, 7, 17], dtype=np.int64)

        # sort
        phi = phi[:, indx]
        d_phi = d_phi[:, indx, :]

        curls = hex.assemble_curl(d_phi)

        # loop over all elements
        for i, e in enumerate(c):
            
            # evaluate this hexahedron for the global position
            points[i*num_pts:(i+1)*num_pts, :] = hex.evaluate_hexahedron(e, self.nodes, phi)

            # compute the Jabcobi matrix
            J = hex.compute_J(e, self.nodes, d_phi)

            # invert the Jacobian
            inv_J = hex.compute_J_inv(J)

            # evaluate the fem solutions partial derivatives in the local coordinates
            dAx_du = d_phi[:, :, 0] @ solution[e, 0]
            dAx_dv = d_phi[:, :, 1] @ solution[e, 0]
            dAx_dw = d_phi[:, :, 2] @ solution[e, 0]

            dAy_du = d_phi[:, :, 0] @ solution[e, 1]
            dAy_dv = d_phi[:, :, 1] @ solution[e, 1]
            dAy_dw = d_phi[:, :, 2] @ solution[e, 1]

            dAz_du = d_phi[:, :, 0] @ solution[e, 2]
            dAz_dv = d_phi[:, :, 1] @ solution[e, 2]
            dAz_dw = d_phi[:, :, 2] @ solution[e, 2]

            # this is transforming the derivatives to the global coorinates
            dAx_dy = dAx_du*inv_J[:, 0, 1] + dAx_dv*inv_J[:, 1, 1] + dAx_dw*inv_J[:, 2, 1]
            dAx_dz = dAx_du*inv_J[:, 0, 2] + dAx_dv*inv_J[:, 1, 2] + dAx_dw*inv_J[:, 2, 2]

            dAy_dx = dAy_du*inv_J[:, 0, 0] + dAy_dv*inv_J[:, 1, 0] + dAy_dw*inv_J[:, 2, 0]
            dAy_dz = dAy_du*inv_J[:, 0, 2] + dAy_dv*inv_J[:, 1, 2] + dAy_dw*inv_J[:, 2, 2]

            dAz_dx = dAz_du*inv_J[:, 0, 0] + dAz_dv*inv_J[:, 1, 0] + dAz_dw*inv_J[:, 2, 0]
            dAz_dy = dAz_du*inv_J[:, 0, 1] + dAz_dv*inv_J[:, 1, 1] + dAz_dw*inv_J[:, 2, 1]

            # this is assembling the curl
            B[i*num_pts:(i+1)*num_pts, 0] = dAz_dy - dAy_dz
            B[i*num_pts:(i+1)*num_pts, 1] = dAx_dz - dAz_dx
            B[i*num_pts:(i+1)*num_pts, 2] = dAy_dx - dAx_dy

        return points, B
    
    def compute_H_in_fem_domain(self, solution=np.array([]), quad_order=1, BH_data=np.zeros((0, 2))):
        '''Evaluate the magnetic flux density in the iron
        domain.
        
        :param solution:
            The solution vector.

        :param quad_order:
            The quad_order defines the number of points to evaluate in each finite element.
            Default = 1.

        :param BH_data:
            The BH data table. If there are no rows, we treat the domain as air.
        
        :return:
            Two numpy arrays with three columns. The first array contains the
            evaluation coordinates. The secon column contains the evaluated vector
            potential coordinates at these positions.
        '''

        # evaluate B
        points, B_field = self.compute_B_in_fem_domain(quad_order=quad_order)

        if BH_data.shape[0] == 0:
            print('No BH data specified! Using air!')
            H_field = B_field/4/np.pi/1e-7
        else:
            # fit to the B(H) data
            t, c, k = interpolate.splrep(BH_data[:, 0], BH_data[:, 1], k=3)

            # construct the spline
            spline = interpolate.BSpline(t, c, k, extrapolate=True)

            # compute the B magnitude
            B_magn = np.linalg.norm(B_field, axis=1)

            # evaluate the spline here
            H_magn = spline(B_magn)

            # compute the H vector
            H_field = B_field.copy()

            # scale it
            H_field[:, 0] *= H_magn/B_magn
            H_field[:, 1] *= H_magn/B_magn
            H_field[:, 2] *= H_magn/B_magn

        return points, H_field

    def get_state_vectors(self, boundary=False):
        '''Get the state vectors. I.e. the magnetic
        vector potential and the normal derivatives
        at the nodal coordinates.

        :param boundary:
            Set this flag to return the boundary data vectors.
        
        :return:
            The numpy arrays of dimension (K, 3), where
            K is the number of nodes in the mesh.
            The first one is for the magnetic vector
            potential components, and the second one is for
            the corresponding normal derivatives.
        '''
        
        # get the magnetic vector potential
        u = sort_solution(self.node_number_list[0], self.pot_list[0])

        # get the normal derivative
        dn_u = sort_solution(self.node_number_list[0], self.der_list[0])

        if boundary:
            b_node_t = self.make_boundary_nodes_table()

            u = u[b_node_t[:, 1] > -1, :]
            dn_u = dn_u[b_node_t[:, 1] > -1, :]

        return u, dn_u
    
    def recover_full_vector(self, vec):
        '''Given a vector with the size matching the number of boundary
        nodes, return a vector with size matching the total number
        of nodes, where the inner nodes are filled with zeros.

        :param vec:
            The vector with the nodal values for the boundary nodes.

        :return:
            The full vector.
        '''
        b_node_t = self.make_boundary_nodes_table()

        mask = b_node_t[:, 1] > -1

        vec_full = np.zeros((b_node_t.shape[0], vec.shape[1]))

        vec_full[mask, :] = vec
        
        return vec_full

    def compute_iron_field(self, points, quad_order=8, field='A', jit=True, u=np.zeros((0, 3)), dn_u=np.zeros((0, 3))):
        '''Evaluate the magnetic vector potential at the
        positions given by the points array.
        This code is for a static magnetic field simulation.
        We read the solution of the last element in the result list.

        :param points:
            The points, i.e. a numpy array of dimension (M x 3),
            where M is the number of evaluation points.

        :param quad_order:
            The order of the quadrature rule.

        :param field:
            A string specifying the field to compute. Options are: 
            "A": magnetic vector potential. "B": Magnetic flux density,
            "all": magnetic vector potential and magnetic flux density.

        :param jit:
            Enables the numba just in time compilation.
            
        :param u:
            A solution vector for the magnetic vector potential. Default np.zeros((0, 3)),
            which means that the loaded solution is used.

        :param dn_u:
            A solution vector for the normal derivatives of the magnetic vector potential.
            Default np.zeros((0, 3)), which means that the loaded solution is used.
        
        :return:
            The field vectors at these positions in an
            (M x 3) numpy array.
        '''

        # number of evaluation points
        num_eval = points.shape[0]

        if u.shape[0] == 0:
            # get the magnetic vector potential
            u = sort_solution(self.node_number_list[0], self.pot_list[0])

        if dn_u.shape[0] == 0:
            # get the normal derivative
            dn_u = sort_solution(self.node_number_list[0], self.der_list[0])

        # make all the interpolations
        q, n, w, A, dA = self.make_all_interpolations(u=u,
                                                      dn_u=dn_u,
                                                      quad_order=quad_order)


        # evaluate
        if field == 'A':
            return compute_A_iron_cpp(points, q, n, w, A, dA)
        elif field == 'B':
            return compute_B_iron_cpp(points, q, n, w, A, dA)
        elif field == 'all':
            A_eval = compute_A_iron_cpp(points, q, n, w, A, dA)
            B_eval = compute_B_iron_cpp(points, q, n, w, A, dA)

            return A_eval, B_eval
        else:
            print('Field {} unknown.'.format(field))
            return 0.0
        
    def compute_inverse_distance(self, point, quad_points):
        '''Compute the inverse distance for a given evaluation point
        and an array of quadrature points.

        :param points:
            The evaluation points Cartesian coordinates.

        :param quad_points:
            An (M x 3) array of quadrature points.

        :return:
            The inverse distance evaluated for these interactions.
        '''

        # the distance vectors
        dx = point[0] - quad_points[:, 0]
        dy = point[1] - quad_points[:, 1]
        dz = point[2] - quad_points[:, 2]

        return 1.0/np.sqrt(dx*dx + dy*dy + dz*dz)
    
        
    def get_number_of_boundary_elements(self):
        '''Get the number of boundary elements.

        :return:
            The number of boundary elements.
        '''
        return len(self.types_b)

    def get_boundary_mesh(self):
        '''Extract the boundary mesh information.

        :return:
            The connectivity and the cell types.
        '''
        # We need to sort the cells differently
        c = self.c_b.copy()

        # the number of elements
        num_elements = c.shape[0]

        # the number of nodes of each element, and the finite element codes for vtk
        num_nodes = np.zeros((len(self.types_b), ), dtype=np.int32)
        vtk_cell_types = np.zeros((len(self.types_b), ), dtype=np.int32)

        # this table is to map between cor file indices and hmo file indices
        node_map = self.node_index_table[:, 1] - 1

        # a counter variable
        cnt = 0

        ele_cnt = 0

        for i in range(len(self.types_b)):

            num_nodes[i] = get_num_nodes_ele(self.types_b[i])
            
            vtk_cell_types[i] = get_vtk_cell_code(num_nodes[i], dim=2)

            if num_nodes[i] == 6:
                stencil = [0, 2, 4, 1, 3, 5]
            elif num_nodes[i] == 8:
                stencil = [0, 2, 4, 6, 1, 3, 5, 7]
                             
            this_c = c[cnt+1:cnt+1+num_nodes[i]] - 1

            c[cnt+1:cnt+1+num_nodes[i]] = node_map[this_c[stencil]]

            cnt += 1+num_nodes[i]

            ele_cnt += 1

        return c, vtk_cell_types
    
    def plot_fem_field(self, pl, field='B', quad_order=1, cmap='jet', mag=1.0, title='Field', BH_data=np.zeros((0, 2)), shift=np.zeros((3, )), opacity=None):
        '''Plot the FEM field in the iron domain.

        :param pl:
            A pyvista plotter object.

        :param field:
            A string specifying which field to plot. 'A', 'B' or 'H'. If 'H' is selected,
            You need to specify also a B(H) data table.

        :param quad_order:
            The quadrature order specifies the number of points to plot in each element.
            Default 1.

        :param cmap:
            A colormap. Default 'jet'.

        :param mag:
            The maximum vector magnitude. Default 1.

        :param title:
            The title for the colormap.

        :param shift:
            A shift in the xyz coordinates.

        :param opacity:
            Set the opacity of the color map. Default = None.
            linear, linear_r, geom, geom_r, sigmoid, sigmoid_r.

        :return:
            None
        '''

        if field == 'A':
            points, field = self.compute_A_in_fem_domain(quad_order=quad_order)
            points, field = self.apply_symmetry(points, field, field='A')
        elif field == 'B':
            points, field = self.compute_B_in_fem_domain(quad_order=quad_order)
            points, field = self.apply_symmetry(points, field, field='B')
        elif field == 'H':
            points, field = self.compute_H_in_fem_domain(quad_order=quad_order, BH_data=BH_data)
            points, field = self.apply_symmetry(points, field, field='B')

        else:
            print('ERROR: Field {} is unknown!'.format(field))
            return None
        
        # the scalars for the arrows
        scalars_field = matlib.repmat(np.linalg.norm(field, axis=1), 15, 1).T.flatten()

        # apply shift
        points[:, 0] += shift[0]
        points[:, 1] += shift[1]
        points[:, 2] += shift[2]

        pl.add_arrows(points, field,
                      mag=mag,
                      cmap=cmap,
                      scalars=scalars_field,
                      scalar_bar_args={"title": title, "color": 'k'},
                      opacity=opacity)
        


        return None

    def plot_boundary_mesh(self, pl, sol=np.zeros((0, )), plot_averages=False):
        '''Plot the boundary mesh in 3D using pyvista.

        :param pl:
            A plotter object.

        :param sol:
            A given solution vector (on all nodes (also internal)).

            
        :param plot_averages:
            A flag, to specify if averages (over the elements) shall be plotted.

        :return:
            The plotter object.
        '''

        # get the boundary mesh
        c, vtk_cell_types = self.get_boundary_mesh()

        # make a pyvista unstructured grid
        mesh = pv.UnstructuredGrid(c, vtk_cell_types, self.nodes)
            
        if sol.shape[0] == 0:
            # add the mvp as node data
            A = sort_solution(self.node_number_list[0], self.pot_list[0])
            A_mag = np.linalg.norm(A, axis=1)

        else:
            A_mag = sol


        if plot_averages:

            # number of boundary elements
            num_boundary_elements = len(vtk_cell_types)

            # the solution vector is coding the faces of the elements
            A_ele = np.zeros((num_boundary_elements, ))

            # a counter for the index in c
            cnt_c = 0

            # loop over the elements
            for i in range(num_boundary_elements):

                # the number of nodes
                num_nodes = c[cnt_c]

                A_ele[i] = np.mean(A_mag[c[cnt_c+1:cnt_c+num_nodes+1]])

                cnt_c += num_nodes + 1


            # add to plot
            pl.add_mesh(mesh, show_edges=True, scalars=A_ele, cmap='jet')

        else:
            # add to plot
            pl.add_mesh(mesh, show_edges=True, scalars=A_mag, cmap='jet')

        # else:
        #     # add to plot
        #     pl.add_mesh(mesh, show_edges=True)            

        return pl
    

    def plot_coil(self, pl, coil_color=[220/255, 80/255, 51/255], shift=np.zeros((3, ))):
        '''Plot the coil geometry 3D using pyvista.

        :param pl:
            A pyvista plotter object.

        :param coil_color:
            The color of the coil (RGB)

        :param shift:
            A shift in the xyz coordinates.

        :return:
            The updated plotter.        

        '''

        points = self.p_cond.copy()

        points[:, 0] += shift[0]
        points[:, 1] += shift[1]
        points[:, 2] += shift[2]

        # make the coil mesh
        cell_info = np.append(np.ones((len(self.c_cond), 1), dtype=np.int32)*8, self.c_cond, axis=1)
        coil_mesh = pv.UnstructuredGrid(cell_info, [pv.CellType.HEXAHEDRON]*self.c_cond.shape[0], points)

        coil_surf = coil_mesh.extract_surface()
        coil_surf.compute_normals(inplace=True, split_vertices=True)

        coil_mesh = pl.add_mesh(coil_surf, show_edges=True, color=coil_color, pbr=True, metallic=0.5, roughness=0.3)

        return pl

    def plot_geometry(self, pl,
                      iron_color='white',
                      coil_color=[220/255, 80/255, 51/255],
                      rot=np.array([0.0, 0.0, 0.0]),
                      transl=np.array([0.0, 0.0, 0.0]),
                      show_edges=False):
        '''Plot the boundary mesh in 3D using pyvista.

        :param pl:
            A plotter object.

        :param iron_color:
            The color of the iron yoke.

        :param coil_color:
            The color of the coil.

        :param rot:
            A vector with three components. One for each possible rotation (x, y, z axis in the
            order R_z*R_y*R_x), in degrees.

        :param transl:
            A vector with three components. One for each displacement in x, y and z.
            Translations are applied after the rotations.

        :return:
            The plotter object.
        '''
        
        # pv.global_theme.silhouette.feature_angle = 80.0

        # make the coil mesh
        cell_info = np.append(np.ones((len(self.c_cond), 1), dtype=np.int32)*8, self.c_cond, axis=1)
        coil_mesh = pv.UnstructuredGrid(cell_info, [pv.CellType.HEXAHEDRON]*self.c_cond.shape[0], self.p_cond)

        coil_surf = coil_mesh.extract_surface()
        coil_surf.compute_normals(inplace=True, split_vertices=True)

        # get the boundary mesh
        c_iron, vtk_cell_types_iron = self.get_boundary_mesh()

        # make a pyvista unstructured grid
        iron_mesh_list = [pv.UnstructuredGrid(c_iron, vtk_cell_types_iron, self.nodes)]

        # apply rotations
        coil_surf.rotate_x(rot[0], inplace=True)
        coil_surf.rotate_y(rot[1], inplace=True)
        coil_surf.rotate_z(rot[2], inplace=True)

        # apply translations
        coil_surf.translate(transl, inplace=True)

        coil_mesh = pl.add_mesh(coil_surf, show_edges=True, color=coil_color, pbr=True, metallic=0.5, roughness=0.3)

        # apply symmetry
        if self.sym_xy > 0:
            iron_mesh_list.append(iron_mesh_list[-1].reflect((0, 0, 1), point=(0, 0, 0)))

        if self.sym_xz > 0:
            for i in range(len(iron_mesh_list)):
                iron_mesh_list.append(iron_mesh_list[i].reflect((0, 1, 0), point=(0, 0, 0)))

        if self.sym_yz > 0:
            for i in range(len(iron_mesh_list)):
                iron_mesh_list.append(iron_mesh_list[i].reflect((1, 0, 0), point=(0, 0, 0)))


        for im in iron_mesh_list:

            # im.save('test.vtk')
            iron_surf = im.extract_surface()
            iron_surf.compute_normals(inplace=True, split_vertices=True)
            # centers = iron_surf.cell_centers().points
            # arrows = iron_surf['Normals']
            edges_iron = iron_surf.extract_feature_edges(45.)
            
            # apply rotations
            iron_surf.rotate_x(rot[0], inplace=True)
            iron_surf.rotate_y(rot[1], inplace=True)
            iron_surf.rotate_z(rot[2], inplace=True)
            edges_iron.rotate_x(rot[0], inplace=True)
            edges_iron.rotate_y(rot[1], inplace=True)
            edges_iron.rotate_z(rot[2], inplace=True)

            # apply translations
            iron_surf.translate(transl, inplace=True)
            edges_iron.translate(transl, inplace=True)

            pl.add_mesh(iron_surf, color=iron_color, pbr=False, show_edges=show_edges, smooth_shading=True, split_sharp_edges=True)
            pl.add_mesh(edges_iron, color='black', line_width=2)
            # pl.add_arrows(centers, arrows, color='black', mag=0.01)



        return pl
    

    def plot_iron_feature_edges(self, pl, linewidth=2, color='k', show_surface=False, surface_color='w', shift=np.zeros((3, ))):
        '''Plot only the feature edges of the iron domain.
        
        :param pl:
            A pyvista plotter object.

        :param linewidth:
            The linewidth of the feature edges.

        :param color:
            The color of the feature edges.

        :param show_surface:
            Set this flag to draw also the surface of the iron air interface.

        :param surface_color:
            The iron air interface is shown in this color if show_surface is selected.
            
        :return:
            None.
        '''

        # get the boundary mesh
        c_iron, vtk_cell_types_iron = self.get_boundary_mesh()


        # copy the nodes
        # nodes = self.nodes.copy()
        
        # apply shift
        # nodes[:, 0] += shift[0]
        # nodes[:, 1] += shift[1]
        # nodes[:, 2] += shift[2]

        # make a pyvista unstructured grid
        iron_mesh_list = [pv.UnstructuredGrid(c_iron, vtk_cell_types_iron, self.nodes)]

        # apply symmetry
        if self.sym_xy > 0:
            iron_mesh_list.append(iron_mesh_list[-1].reflect((0, 0, 1), point=(0, 0, 0)))

        if self.sym_xz > 0:
            for i in range(len(iron_mesh_list)):
                iron_mesh_list.append(iron_mesh_list[i].reflect((0, 1, 0), point=(0, 0, 0)))

        if self.sym_yz > 0:
            for i in range(len(iron_mesh_list)):
                iron_mesh_list.append(iron_mesh_list[i].reflect((1, 0, 0), point=(0, 0, 0)))

        for im in iron_mesh_list:
            im.translate(shift, inplace=True)
            iron_surf = im.extract_surface()
            iron_surf.compute_normals(inplace=True, split_vertices=True)
            edges_iron = iron_surf.extract_feature_edges(45.)

            

            pl.add_mesh(edges_iron, color=color, line_width=linewidth)

            if show_surface:
                # iron_surf.translate(shift, inplace=True)
                pl.add_mesh(iron_surf, color=surface_color)

        return None
        
    def plot_solution(self, pl,
                      sol=np.zeros((0, 3)),
                      show_edges=False,
                      save=False,
                      filename='sol',
                      cmap='jet',
                      feature_edges=False,
                      clim=[]):
        '''Plot the solution in 3D using pyvista.

        :param pl:
            A plotter object.

        :param sol:
            The solution as an M x 3 array, where M is the number of
            Dofs. Default is np.zeros((0, 3)), that means the read solution
            is plotted.

        :param show_edges:
            Set this flag if You want to show the edges.

        :param save:
            Set this flag to generate a vtk output.

        :param filename:
            Specify the filename of the vtk output. Ignored if save is False.
            The extension vtk is appended.

        :param cmap:
            A color map. Default 'jet'.

        :param feature_edges:
            Set this flag to show feature edges.

        :param clim:
            Set manually the colorbar range. Default [] which means it is adjusted
            according to the data.

        :return:
            The plotter object.
        '''

        # We need to sort the cells differently
        c = self.elements.copy()
        c -= 1

        # the number of elements
        num_elements = self.elements.shape[0]

        # the number of nodes of each element, and the finite element codes for vtk
        num_nodes = np.zeros((len(self.element_codes), 1), dtype=np.int32)
        vtk_cell_types = np.zeros((len(self.element_codes), 1), dtype=np.int32)

        cells_info = np.zeros((0, ), dtype=np.int32)

        for i in range(len(self.element_codes)):
            num_nodes[i] = get_number_of_nodes(self.element_codes[i])
            vtk_cell_types[i] = get_vtk_cell_code(num_nodes[i])

            if num_nodes[i] == 20:
                stencil = [0, 2, 4, 6, 12, 14, 16, 18, 1, 3, 5, 7, 13, 15, 17, 19, 8, 9, 10, 11]
            elif num_nodes[i] == 15:
                stencil = [0, 2, 4, 9, 11, 13, 1, 3, 5, 10, 12, 14, 6, 7, 8]
                
            c[i, :num_nodes[i, 0]] = c[i, stencil]

            this_cell_info = np.zeros((num_nodes[0, 0]+1, ), dtype=np.int32)
            this_cell_info[0] = num_nodes[0]
            this_cell_info[1:] = c[i, :] 
            
            cells_info = np.append(cells_info, this_cell_info)
            
        iron_mesh_list = [pv.UnstructuredGrid(cells_info,
                                    vtk_cell_types,
                                    self.nodes)]
        
        # apply symmetry
        if self.sym_xy > 0:
            iron_mesh_list.append(iron_mesh_list[-1].reflect((0, 0, 1), point=(0, 0, 0)))

        if self.sym_xz > 0:
            for i in range(len(iron_mesh_list)):
                iron_mesh_list.append(iron_mesh_list[i].reflect((0, 1, 0), point=(0, 0, 0)))

        if self.sym_yz > 0:
            for i in range(len(iron_mesh_list)):
                iron_mesh_list.append(iron_mesh_list[i].reflect((1, 0, 0), point=(0, 0, 0)))


        if sol.shape[0] == 0:
            A = sort_solution(self.node_number_list[0], self.pot_list[0])
            A_mag = np.linalg.norm(A, axis=1)

        else:
            A_mag = sol

        mesh_list = []
        for i, iron_mesh in enumerate(iron_mesh_list):
            if len(clim) == 2:
                mesh_list.append(pl.add_mesh(iron_mesh, show_edges=show_edges, scalars=A_mag, cmap=cmap, edge_color='gray', clim=clim,
                                 scalar_bar_args={"title": "|A| in Tm", "color": 'k'}))
            else:
                mesh_list.append(pl.add_mesh(iron_mesh, show_edges=show_edges, scalars=A_mag, cmap=cmap, edge_color='gray',
                                 scalar_bar_args={"title": "|A| in Tm", "color": 'k'}))
            if feature_edges:
                iron_surf = iron_mesh.extract_surface()
                iron_surf.compute_normals(inplace=True, split_vertices=True)
                # centers = iron_surf.cell_centers().points
                # arrows = iron_surf['Normals']
                edges_iron = iron_surf.extract_feature_edges(45.)

                pl.add_mesh(edges_iron, color='white', line_width=2)

            if save:
                iron_mesh.save(filename + '_{}'.format(i) + '.vtk')

        return pl
    
    def get_solution(self):
        '''Get the solution arrays u and dn_u.
        
        :return:
            The roxie solution numpy arrays.

        '''
        # get the magnetic vector potential
        u = sort_solution(self.node_number_list[0], self.pot_list[0])

        # get the normal derivative
        dn_u = sort_solution(self.node_number_list[0], self.der_list[0])

        return u, dn_u

    def make_design_matrix(self, points,
                                 quad_order=8,
                                 type='double_layer'):
        '''Make the design matrix for the B field evaluation of a magnetic single
        or double layer potential.
        
        :param points:
            The points where to evaluate the field.
    
        :param quad_order:
            The order of the quadrature rule.

        :param type:
            The potential type. Either "single_layer' or 'double_layer'.

        :return:
            The three design matrices. These matrices are of dimension (3*P x N)
            where P is the number of evaluation points, and N is the number of
            boundary nodes.
            Each matrix corresponds to one component of the magnetic vector potential.
            The product y = H_x @ u, where u is a solution vector for the Ax component,
            gives the B field vectors at the evaluation points, whereas we append the
            components of B in along the rows, i.e.
            y = (B_x(r_1), B_x(r_2), ..., B_x(r_P), B_y(r_1), ..., B_z(r_P))^T.
        '''

        print('WARNING: This code currently ignores any symmetry condition!')

        pl = pv.Plotter()
        self.plot_geometry(pl)
        pl.add_mesh(points, point_size=5.0, render_points_as_spheres=True)
        pl.show()

        # make the boundary nodes table
        b_node_table = self.make_boundary_nodes_table()

        # count the number of boundary nodes
        num_b_nodes = b_node_table[b_node_table[:, 1] > -1, :].shape[0]

        # the number of evaluation points
        num_eval = points.shape[0]

        # make space for the return matrices, we compute three of them, one for each component
        # of the vector potential
        H_x = np.zeros((3*num_eval, num_b_nodes))
        H_y = np.zeros((3*num_eval, num_b_nodes))
        H_z = np.zeros((3*num_eval, num_b_nodes))

        # get the number of boundary elements
        num_el = self.get_number_of_boundary_elements()

        # get the boundary mesh
        c, _ = self.get_boundary_mesh()

        # make T6 and Q8 elements
        t6 = triangle_T6()
        q8 = quad_Q8()

        # get the quadrature rules
        w_T6, q_T6 = t6.get_quadrarure_rule(quad_order)
        w_Q8, q_Q8 = q8.get_quadrarure_rule(quad_order)

        # evaluate the basis functions
        N_T6 = t6.evaluate_basis(q_T6)
        N_Q8 = q8.evaluate_basis(q_Q8)
        

        # and also the derivatives
        d_N_T6 = t6.evaluate_basis_derivative(q_T6)
        d_N_Q8 = q8.evaluate_basis_derivative(q_Q8)

        # a counter for the indices in the element list
        cnt_c = 0

        for i in tqdm(range(num_el)):

            # the number of nodes
            num_nodes = c[cnt_c]

            if num_nodes == 6:


                # quadrature points
                q = t6.evaluate(c[cnt_c+1:cnt_c+7], self.nodes, N_T6)

                # surface elements and normal vectors
                surf_el, n = t6.evaluate_surface_element(c[cnt_c+1:cnt_c+7], self.nodes, d_N_T6)

                # evaluate the integral kernels (is an P x M array, where P is the number of evaluation points)
                kernel_x, kernel_y, kernel_z = self.evaluate_curl_dl_kernel(points, q, n) 

                # TO DO: Apply symmetry!!!

                # loop over all basis functions
                for j in range(6):

                    # the index of this boundary node
                    indx = b_node_table[c[cnt_c+1+j], 1]

                    # compute the interaction
                    H_x[:, indx] += kernel_x @ (surf_el * w_T6 * N_T6[:, j])
                    H_y[:, indx] += kernel_y @ (surf_el * w_T6 * N_T6[:, j])
                    H_z[:, indx] += kernel_z @ (surf_el * w_T6 * N_T6[:, j])

            elif num_nodes == 8:

                # quadrature points
                q = q8.evaluate(c[cnt_c+1:cnt_c+9], self.nodes, N_Q8)

                # surface elements and normal vectors
                surf_el, n = q8.evaluate_surface_element(c[cnt_c+1:cnt_c+9], self.nodes, d_N_Q8)

                # evaluate the integral kernels (is an P x M array, where P is the number of evaluation points)
                kernel_x, kernel_y, kernel_z = self.evaluate_curl_dl_kernel(points, q, n) 

                # TO DO: Apply symmetry!!!

                # loop over all basis functions
                for j in range(8):

                    # the index of this boundary node
                    indx = b_node_table[c[cnt_c+1+j], 1]

                    # compute the interaction
                    H_x[:, indx] += kernel_x @ (surf_el * w_Q8 * N_Q8[:, j])
                    H_y[:, indx] += kernel_y @ (surf_el * w_Q8 * N_Q8[:, j])
                    H_z[:, indx] += kernel_z @ (surf_el * w_Q8 * N_Q8[:, j])


            else:
                print('No element with {} nodes known!'.format(num_nodes))

            cnt_c += num_nodes + 1
            
        return H_x, H_y, H_z

    def evaluate_curl_dl_kernel(self, points, q, n):
        '''Evaluate an integral kernel.
        
        :param points:
            The target points.

        :param q:
            The source points.

        :param n:
            The source normal vectors.

        :return:
            A numpy matrix of size (P x M) where P is the number of
            target points and M is the number of source points.
        ''' 

        # the number of source points
        M = q.shape[0]

        # the number of target points
        P = points.shape[0]

        # compute the distance vectors
        diff = np.zeros((P, M, 3))

        # the products n*diff
        nd = np.zeros((P, M))

        # fill them
        for m in range(M):
            diff[:, m, 0] = points[:, 0] - q[m, 0]
            diff[:, m, 1] = points[:, 1] - q[m, 1]
            diff[:, m, 2] = points[:, 2] - q[m, 2]

            nd[:, m] = diff[:, m, 0]*n[m, 0] + diff[:, m, 1]*n[m, 1] + diff[:, m, 2]*n[m, 2]

        # the distances
        dist = np.linalg.norm(diff, axis=2)
        # the distance to the powers of 3 and 5
        dist_3 = dist*dist*dist
        dist_5 = dist_3*dist*dist
        
        # the two parts of the kernel
        f2 = np.zeros((P, M, 3))

        # fill them
        for m in range(M):

            f2[:, m, 0] = (3.0*nd[:, m]*diff[:, m, 0]/dist_5[:, m] - n[m, 0]/dist_3[:, m])/4.0/np.pi
            f2[:, m, 1] = (3.0*nd[:, m]*diff[:, m, 1]/dist_5[:, m] - n[m, 1]/dist_3[:, m])/4.0/np.pi
            f2[:, m, 2] = (3.0*nd[:, m]*diff[:, m, 2]/dist_5[:, m] - n[m, 2]/dist_3[:, m])/4.0/np.pi

        # The kernel evaluation for the three vector components 
        kernel_x = np.zeros((3*P, M))
        kernel_y = np.zeros((3*P, M))
        kernel_z = np.zeros((3*P, M))
        
        # kernel_x[:P, :] = 0.0
        kernel_x[P:2*P, :] = -1.0*f2[:, :, 2]
        kernel_x[2*P:, :] = f2[:, :, 1]

        kernel_y[:P, :] = f2[:, :, 2]
        # kernel_y[P:2*P, :] = 0.0
        kernel_y[2*P:, :] = -f2[:, :, 0]

        kernel_z[:P, :] = -1.0*f2[:, :, 1]
        kernel_z[P:2*P, :] = f2[:, :, 0]
        # kernel_z[2*P:, :] = 0.0


        return kernel_x, kernel_y, kernel_z

    def make_integral_multipoles_design_matrix(self, z_0, z_1, num_z, N, r_ref, num_phi=64, x_0=0.0, y_0=0.0):
        '''Make a design matrix for integral multipole measurements at positions along the
        z axis.

        :param z_0:
            The z initial position.

        :param z_1:
            The z final position.

        :param num_z:
            The number of steps along z.
                 
        :param N:
            The number of multipoles to reconstruct.

        :param r_ref:
            The reference radius.
            
        :param num_phi:
            The number of angular steps to approximate the multipoles.
            Default 64.
            
        :param x_0:
            The horizontal position on which the multipoles are expressed.

        :param y_0:
            The vertical position on which the multipoles are expressed.
            
        :return:
            Six numpy matrices for the evaluation of the normal components based on the three
            components of the magnetic vector potential.
        '''

        H_b_x, H_b_y, H_b_z, H_a_x, H_a_y, H_a_z = self.make_multipoles_design_matrix(z_0,
                                                                                      z_1,
                                                                                      num_z,
                                                                                      N,
                                                                                      r_ref,
                                                                                      num_phi=num_phi,
                                                                                      x_0=x_0,
                                                                                      y_0=x_0)

        # Matrix to compute the integral multipoles
        # ====
        # the stepsize

        # the discretization steps
        z_pos = np.linspace(z_0, z_1, num_z)

        dz = np.mean(np.diff(z_pos))

        # the stamp to apply
        stamp = dz*np.ones((num_z, ))
        stamp[0] *= 0.5
        stamp[-1] *= 0.5

        M_int = np.zeros((N, N*num_z))

        for i, nn in enumerate(range(1, N+1)):
            M_int[i, i*num_z:(i+1)*num_z] = stamp

        
        # overall matrices
        H_b_x = M_int @ H_b_x
        H_a_x = M_int @ H_a_x

        H_b_y = M_int @ H_b_y
        H_a_y = M_int @ H_a_y

        H_b_z = M_int @ H_b_z
        H_a_z = M_int @ H_a_z

        return H_b_x, H_b_y, H_b_z, H_a_x, H_a_y, H_a_z
    

    def make_multipoles_design_matrix(self, z_0, z_1, num_z, N, r_ref, num_phi=64, x_0=0.0, y_0=0.0):
        '''Make a design matrix for multipole measurements at positions along the
        z axis.

        :param z_0:
            The z initial position.

        :param z_1:
            The z final position.

        :param num_z:
            The number of steps along z.
                 
        :param N:
            The number of multipoles to reconstruct.

        :param r_ref:
            The reference radius.
            
        :param num_phi:
            The number of angular steps to approximate the multipoles.
            Default 64.
            
        :param x_0:
            The horizontal position on which the multipoles are expressed.

        :param y_0:
            The vertical position on which the multipoles are expressed.
            
        :return:
            Six numpy matrices for the evaluation of the normal and skew components based on the three
            components of the magnetic vector potential.
        '''

        # the discretization steps
        z_pos = np.linspace(z_0, z_1, num_z)

        # the total number of field evaluations
        num_eval = num_phi*num_z

        # make an arrary with all points
        points_all = np.zeros((num_eval, 3))
        phi_all = np.zeros((num_eval, ))
        
        # make a phi vector
        phi = np.linspace(0, 2*np.pi, num_phi, endpoint=False)

        # make the x and y points
        x = r_ref*np.cos(phi) + x_0
        y = r_ref*np.sin(phi) + y_0

        for i, zz in enumerate(z_pos):

            points_all[i*num_phi:(i+1)*num_phi, 0] = x
            points_all[i*num_phi:(i+1)*num_phi, 1] = y
            points_all[i*num_phi:(i+1)*num_phi, 2] = zz

            phi_all[i*num_phi:(i+1)*num_phi] = phi

        # compute the design matrix for B
        H_x, H_y, H_z = self.make_design_matrix(points_all)

        # Matrix to reduce Br
        # ====
        M_Br = np.zeros((num_eval, 3*num_eval))

        for i in range(num_z):
            for j in range(num_phi):
                
                M_Br[i*num_phi + j, i*num_phi + j] = np.cos(phi[j])
                M_Br[i*num_phi + j, i*num_phi + j + num_eval] = np.sin(phi[j])

        # Matrix to compute the multipoles over z
        # ====
        M_Bn_z = np.zeros((N*num_z, num_eval))
        M_An_z = np.zeros((N*num_z, num_eval))

        for i, nn in enumerate(range(1, N+1)):
            for j in range(num_z):
                M_Bn_z[i*num_z + j, j*num_phi:(j+1)*num_phi] = 2.0/num_phi*np.sin(nn*phi)
                M_An_z[i*num_z + j, j*num_phi:(j+1)*num_phi] = 2.0/num_phi*np.cos(nn*phi)

        # overall matrices
        H_b_x = M_Bn_z @ M_Br @ H_x
        H_a_x = M_An_z @ M_Br @ H_x

        H_b_y = M_Bn_z @ M_Br @ H_y
        H_a_y = M_An_z @ M_Br @ H_y

        H_b_z = M_Bn_z @ M_Br @ H_z
        H_a_z = M_An_z @ M_Br @ H_z

        return H_b_x, H_b_y, H_b_z, H_a_x, H_a_y, H_a_z
    

    def evaluate_multipoles(self,
                            z_0,
                            z_1,
                            num_z,
                            N,
                            r_ref,
                            num_phi=64,
                            x_0=0.0,
                            y_0=0.0,
                            quad_order=8,
                            strands=(1, 1),
                            coil_field=True,
                            u=np.zeros((0, 3)),
                            dn_u=np.zeros((0, 3))):
        '''Evaluate the multipole fields along the z axis.

        :param z_0:
            The z initial position.

        :param z_1:
            The z final position.

        :param num_z:
            The number of steps along z.
                 
        :param N:
            The number of multipoles to reconstruct.

        :param r_ref:
            The reference radius.
            
        :param num_phi:
            The number of angular steps to approximate the multipoles.
            Default 64.
            
        :param x_0:
            The horizontal position on which the multipoles are expressed.

        :param y_0:
            The vertical position on which the multipoles are expressed.

        :param quad_order:
            The order of the quadrature rule.
            
        :param strands:
            The strands tuple (M x N) strands are used for conductors.

        :param coil_field:
            Set this flag to add the coil field. Default True.

        :param u:
            A solution vector for the magnetic vector potential. Default np.zeros((0, 3)),
            which means that the loaded solution is used.

        :param dn_u:
            A solution vector for the normal derivatives of the magnetic vector potential.
            Default np.zeros((0, 3)), which means that the loaded solution is used.

        :return:
            A numpy matrix of dimension (K x 2*N) where K is the number of
            z positions and N is the number of multipoles.
            The first N columns are the normal multipoles. The second N
            columns are the skew ones.
            Return also the z positions.
        '''

        # the discretization steps
        z_pos = np.linspace(z_0, z_1, num_z)

        # make an arrary with all points
        points = np.zeros((num_phi, 3))
        phi = np.zeros((num_phi, ))
        
        # make a phi vector
        phi = np.linspace(0, 2*np.pi, num_phi, endpoint=False)

        # make the x and y points
        points[:, 0] = r_ref*np.cos(phi) + x_0
        points[:, 1] = r_ref*np.sin(phi) + y_0


        # the return data
        ret_data = np.zeros((num_z, 2*N))

        # loop over the points
        for i, zz in enumerate(z_pos):

            # set the z position
            points[:, 2] = zz

            # evaluate iron field
            B_i = self.compute_iron_field(points, quad_order=quad_order, field='B', u=u, dn_u=dn_u)

            if coil_field:
                # evaluate coil field
                B_c = self.compute_coil_field(points, strands[0], strands[1], field='B')

                # combine them
                B = B_i + B_c

            else:
                B = B_i

            # get Br
            Br = B[:, 0]*np.cos(phi) + B[:, 1]*np.sin(phi)

            # compute the multipoles
            for n in range(1, N+1):
                ret_data[i, n - 1] = 2.0/num_phi*np.sum(np.sin(n*phi)*Br)
                ret_data[i, N + n - 1] = 2.0/num_phi*np.sum(np.cos(n*phi)*Br)


        return ret_data, z_pos

    def compute_B(self, positions, Nn, Nb, quad_order=10, solution=np.zeros((0, )), run_gmsh_gui=False, near_field_ratio=0.5):
        '''Compute the B field (interior or exterior) at certain posisions.

        :param positions:
            A numpy array of size (M x 3) with the coordinates in the columns.

        :param Nn:
            Number of filaments in N direction.

        :param Nb:
            Number of filaments in B direction.
            
        :param quad_order:
            The order of the numerical quadrature.

        :param solution:
            A solution vector. If len(solution) == 0 (default), then the ROXIE
            solution vector is used.

        :param run_gmsh_gui:
            Set this flag if You like to run the gmsh gui.

        :param near_field_ratio:
            The near_field_ratio. All interactions which are closer than near_field_ratio*diam
            will be considered near field, where diam is the domain diameter.

        :return:
            A numpy matrix of dimension (M x 3) with the B field components in the
            columns. 
        '''

        # the number of positions
        num_pos = positions.shape[0]

        # the number of nodes
        num_nodes = self.nodes.shape[0]

        # we make use of gmsh to determine if a point is inside the mesh or outside.
        # we also use it to get the parametric coordinates of the finite element.
        # we therefore need to translate between ROXIE and gmsh.

        # we initialize gmsh if not already done
        if not gmsh.isInitialized():
            gmsh.initialize()

        # clear everything
        gmsh.clear()

        # clear everything
        gmsh.model.occ.synchronize()

        # we make a new volume entity (check if we need to split the domains here!)
        vol = gmsh.model.addDiscreteEntity(3)

        # we add the nodes to the volume
        gmsh.model.mesh.add_nodes(3, vol,
                                    np.linspace(1, num_nodes, num_nodes),
                                    self.nodes.flatten())


        # synchronization neccessary?     
        gmsh.model.occ.synchronize()

        # this is to sort our finite element nodes to the gmsh definition
        sorting = [0, 2, 4, 6, 12, 14, 16, 18, 1, 7, 8, 3, 9, 5, 10, 11, 13, 19, 15, 17]

        # we add all the finite elements 
        gmsh.model.mesh.addElementsByType(vol, 17,
                                  np.linspace(1, self.elements.shape[0], self.elements.shape[0]),
                                  self.elements[:, sorting].flatten())
        
        # check if this detects also disjoint volumes 
        gmsh.model.mesh.createTopology()

        # generate the mesh
        gmsh.model.mesh.generate(3)

        if run_gmsh_gui:
            gmsh.fltk.run()

        # thats it, the gmsh mesh can now be evaluated!

        # check if solution vector is given
        if len(solution) == 0:
            solution, _ = self.get_solution()
            solution = sort_solution(self.node_number_list[0], self.pot_list[0])

        # the gmsh node definition is different to the one used in hypermesh
        indx = np.array([0, 8, 1, 12, 5, 16, 4, 10, 9, 11, 18, 17, 3, 13, 2, 14, 6, 19, 7, 15], dtype=np.int64)

        # we now allocate a return vector
        B = np.zeros((num_pos, 3))

        # this is a mask for the outside points
        outside_mask = np.zeros((num_pos, ), dtype=bool)

        # for some reason, gmsh renumbers nodes...
        # we need to make a mapping between gmsh node tags and my node tags
        node_map = np.zeros((num_nodes, ), dtype=np.int64)

        # get all volumes
        volumes = gmsh.model.get_entities(3)


        # loop over all nodes
        for i in range(num_nodes):
            # get the coordinates
            coord, _, _, _ = gmsh.model.mesh.get_node(i+1)
            # search for this nodes in my nodes
            diff_x = coord[0] - self.nodes[:, 0]
            diff_y = coord[1] - self.nodes[:, 1]
            diff_z = coord[2] - self.nodes[:, 2]
            # the distance
            dist = np.sqrt(diff_x*diff_x + diff_y*diff_y + diff_z*diff_z)
            # get the minimum
            node_map[i] = np.argmin(dist)

        # we loop over all of the positions
        for i in range(num_pos):


            for v in volumes:
                
                # check if this point is inside or outside
                is_inside = gmsh.model.is_inside(3, v[1], positions[i, :])

                if is_inside == True:
                    break

            if is_inside == 1:

                # get the element and local coordinates
                el, type, node_tags, u, v, w = gmsh.model.mesh.get_element_by_coordinates(positions[i, 0], positions[i, 1], positions[i, 2])

                # the node indices of my numbering
                node_indx = node_map[node_tags-1]

                # evaluate the derivatives at these points
                d_phi = hex.eval_gradient_hexahedron(np.array([[u, v, w]]), 2)
                # d_phi = d_phi[:, indx, :]

                # compute the Jabcobi matrix
                J = hex.compute_J(node_indx, self.nodes, d_phi)
                
                # invert the Jacobian
                inv_J = hex.compute_J_inv(J)

                # evaluate the fem solutions partial derivatives in the local coordinates
                dAx_du = d_phi[:, :, 0] @ solution[node_indx, 0]
                dAx_dv = d_phi[:, :, 1] @ solution[node_indx, 0]
                dAx_dw = d_phi[:, :, 2] @ solution[node_indx, 0]

                dAy_du = d_phi[:, :, 0] @ solution[node_indx, 1]
                dAy_dv = d_phi[:, :, 1] @ solution[node_indx, 1]
                dAy_dw = d_phi[:, :, 2] @ solution[node_indx, 1]

                dAz_du = d_phi[:, :, 0] @ solution[node_indx, 2]
                dAz_dv = d_phi[:, :, 1] @ solution[node_indx, 2]
                dAz_dw = d_phi[:, :, 2] @ solution[node_indx, 2]

                # this is transforming the derivatives to the global coorinates
                dAx_dy = dAx_du*inv_J[:, 0, 1] + dAx_dv*inv_J[:, 1, 1] + dAx_dw*inv_J[:, 2, 1]
                dAx_dz = dAx_du*inv_J[:, 0, 2] + dAx_dv*inv_J[:, 1, 2] + dAx_dw*inv_J[:, 2, 2]

                dAy_dx = dAy_du*inv_J[:, 0, 0] + dAy_dv*inv_J[:, 1, 0] + dAy_dw*inv_J[:, 2, 0]
                dAy_dz = dAy_du*inv_J[:, 0, 2] + dAy_dv*inv_J[:, 1, 2] + dAy_dw*inv_J[:, 2, 2]

                dAz_dx = dAz_du*inv_J[:, 0, 0] + dAz_dv*inv_J[:, 1, 0] + dAz_dw*inv_J[:, 2, 0]
                dAz_dy = dAz_du*inv_J[:, 0, 1] + dAz_dv*inv_J[:, 1, 1] + dAz_dw*inv_J[:, 2, 1]

                # this is assembling the curl
                B[i, 0] = dAz_dy - dAy_dz
                B[i, 1] = dAx_dz - dAz_dx
                B[i, 2] = dAy_dx - dAx_dy
                
            else:

                # mark this point as outside
                outside_mask[i] = True

        if sum(outside_mask) > 0:
                
            # we now evaluate the outside points
            B_c = self.compute_coil_field(positions[outside_mask, :],
                                        Nn, Nb, field='B', near_field_ratio=near_field_ratio)

            B_i = self.compute_iron_field(positions[outside_mask, :], quad_order=quad_order, field='B')
            
            B[outside_mask, :] = B_c + B_i

        return B
    

    def compute_A(self, positions, Nn, Nb, quad_order=10, solution=np.zeros((0, )), run_gmsh_gui=False, near_field_ratio=0.5):
        '''Compute the magnetic vector potential (interior or exterior) at certain posisions.

        :param positions:
            A numpy array of size (M x 3) with the coordinates in the columns.

        :param Nn:
            Number of filaments in N direction.

        :param Nb:
            Number of filaments in B direction.
            
        :param quad_order:
            The order of the numerical quadrature.

        :param solution:
            A solution vector. If len(solution) == 0 (default), then the ROXIE
            solution vector is used.

        :param run_gmsh_gui:
            Set this flag if You like to run the gmsh gui.

        :param near_field_ratio:
            The near_field_ratio. All interactions which are closer than near_field_ratio*diam
            will be considered near field, where diam is the domain diameter.

        :return:
            A numpy matrix of dimension (M x 3) with the B field components in the
            columns. 
        '''

        # the number of positions
        num_pos = positions.shape[0]

        # the number of nodes
        num_nodes = self.nodes.shape[0]

        # we make use of gmsh to determine if a point is inside the mesh or outside.
        # we also use it to get the parametric coordinates of the finite element.
        # we therefore need to translate between ROXIE and gmsh.

        # we initialize gmsh if not already done
        if not gmsh.isInitialized():
            gmsh.initialize()

        gmsh.model.occ.synchronize()

        # clear everything
        gmsh.clear()

        # we make a new volume entity (check if we need to split the domains here!)
        vol = gmsh.model.addDiscreteEntity(3)

        # we add the nodes to the volume
        gmsh.model.mesh.add_nodes(3, vol,
                                    np.linspace(1, num_nodes, num_nodes),
                                    self.nodes.flatten())


        # synchronization neccessary?     
        gmsh.model.occ.synchronize()

        # this is to sort our finite element nodes to the gmsh definition
        sorting = [0, 2, 4, 6, 12, 14, 16, 18, 1, 7, 8, 3, 9, 5, 10, 11, 13, 19, 15, 17]

        # we add all the finite elements 
        gmsh.model.mesh.addElementsByType(vol, 17,
                                  np.linspace(1, self.elements.shape[0], self.elements.shape[0]),
                                  self.elements[:, sorting].flatten())
        
        # check if this detects also disjoint volumes 
        gmsh.model.mesh.createTopology()

        # generate the mesh
        gmsh.model.mesh.generate(3)

        if run_gmsh_gui:
            gmsh.fltk.run()

        # thats it, the gmsh mesh can now be evaluated!

        # check if solution vector is given
        if len(solution) == 0:
            solution, _ = self.get_solution()
            solution = sort_solution(self.node_number_list[0], self.pot_list[0])

        # the gmsh node definition is different to the one used in hypermesh
        indx = np.array([0, 8, 1, 12, 5, 16, 4, 10, 9, 11, 18, 17, 3, 13, 2, 14, 6, 19, 7, 15], dtype=np.int64)
        # indx = np.array([0, 2, 4, 6, 12, 14, 16, 18, 1, 7, 8, 3, 9, 5, 10, 11, 13, 19, 15, 17], dtype=np.int64)

        # we now allocate a return vector
        A = np.zeros((num_pos, 3))

        # this is a mask for the outside points
        outside_mask = np.zeros((num_pos, ), dtype=bool)

        # for some reason, gmsh renumbers nodes...
        # we need to make a mapping between gmsh node tags and my node tags
        node_map = np.zeros((num_nodes, ), dtype=np.int64)

        # loop over all nodes
        for i in range(num_nodes):
            # get the coordinates
            coord, _, _, _ = gmsh.model.mesh.get_node(i+1)
            # search for this nodes in my nodes
            diff_x = coord[0] - self.nodes[:, 0]
            diff_y = coord[1] - self.nodes[:, 1]
            diff_z = coord[2] - self.nodes[:, 2]
            # the distance
            dist = np.sqrt(diff_x*diff_x + diff_y*diff_y + diff_z*diff_z)
            # get the minimum
            node_map[i] = np.argmin(dist)

        # we loop over all of the positions
        for i in range(num_pos):

            # check if this point is inside or outside
            is_inside = gmsh.model.is_inside(3, vol, positions[i, :])

            if is_inside == 1:

                # get the element and local coordinates
                el, type, node_tags, u, v, w = gmsh.model.mesh.get_element_by_coordinates(positions[i, 0], positions[i, 1], positions[i, 2])

                # the node indices of my numbering
                node_indx = node_map[node_tags-1]

                # evaluate the derivatives at these points
                phi = hex.eval_shape_hexahedron(np.array([[u, v, w]]), 2)

                # evaluate this hexahedron for the global position
                val_point = hex.evaluate_hexahedron(node_indx, self.nodes, phi)
                
                # evaluate the fem solution
                A[i, 0] = phi @ solution[node_indx, 0]
                A[i, 1] = phi @ solution[node_indx, 1]
                A[i, 2] = phi @ solution[node_indx, 2]
                
            else:

                # mark this point as outside
                outside_mask[i] = True

        if sum(outside_mask) > 0:

            # we now evaluate the outside points
            A_c = self.compute_coil_field(positions[outside_mask, :],
                                        Nn, Nb, field='A', near_field_ratio=near_field_ratio)

            A_i = self.compute_iron_field(positions[outside_mask, :], quad_order=quad_order, field='A')
            
            A[outside_mask, :] = A_c + A_i

        return A