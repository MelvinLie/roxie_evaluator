# distutils: sources = [cpp/c-algorithms/ctools.cpp, cpp/c-algorithms/evaluators.cpp, cpp/c-algorithms/convertors.cpp, cpp/c-algorithms/analytical.cpp, cpp/c-algorithms/mesh_tools.cpp, cpp/c-algorithms/inductance_calculation.cpp, cpp/c-algorithms/SolidHarmonics/solid_harmonics.cpp, cpp/c-algorithms/boundary_elements.cpp, cpp/c-algorithms/MLFMM/ClusterTree.cpp, cpp/c-algorithms/MLFMM/ClusterTreeMemory.cpp, cpp/c-algorithms/MLFMM/ClusterTreeNode.cpp, cpp/c-algorithms/MLFMM/MultipoleMomentContainer.cpp, cpp/c-algorithms/MLFMM/MagneticFluxDensityMonitor.cpp, cpp/c-algorithms/MLFMM/MagneticVectorPotentialMonitor.cpp, cpp/c-algorithms/MLFMM/VectorSingleLayerMonitor.cpp, cpp/c-algorithms/MLFMM/VectorDoubleLayerMonitor.cpp, cpp/c-algorithms/MLFMM/CurlVectorDoubleLayerMonitor.cpp, cpp/c-algorithms/MLFMM/CurlVectorSingleLayerMonitor.cpp] 
# distutils: include_dirs = [cpp/c-algorithms/, cpp/c-algorithms/SolidHarmonics, cpp/c-algorithms/MLFMM, cpp/c-algorithms/eigen3]
# distutils: language = c++

import numpy as np
cimport numpy as np
cimport ctools
from cython cimport view
from libc.stdlib cimport malloc, free
from scipy.sparse import csr_array
import copy


def compute_B_line_segs_cpp(src, tar, current):
    '''Launch the C++ code to compute the B field for given
    sources, targets and current.

    :param src:
        The sources in an N x 6 numpy array.

    :param tar:
        The targets in an M x 3 numpy array.

    :param current:
        The strand current.

    :return:
        The B fields in an M x 3 numpy array. 
    '''

    # get the number of sources
    num_src = src.shape[0]
    
    if src.shape[1] != 6:
        print("Error! The source array must be of dimension N x 6!")
        return -1 
    
    # get the number of targets
    num_tar = tar.shape[0]
    if tar.shape[1] != 3:
        print("Error! The target array must be of dimension M x 3!")
        return -1 

    # ====================================
    # Convert python -> C
    # ====================================


    # make c type data pointers
    cdef np.ndarray[double, ndim=1, mode = 'c'] src_buff = np.ascontiguousarray(src.flatten(), dtype = np.double)
    cdef double* src_c = <double*> src_buff.data

    cdef np.ndarray[double, ndim=1, mode = 'c'] tar_buff = np.ascontiguousarray(tar.flatten(), dtype = np.double)
    cdef double* tar_c = <double*> tar_buff.data
        
    # ====================================
    # run cpp code
    # ====================================

    cdef double *B_c = <double *> malloc(3*num_tar*sizeof(double))
    ctools.compute_B_line_segs(B_c, src_c, tar_c, current, num_src, num_tar)


    # ===========================================
    # Convert C -> python
    # ===========================================
    
    cdef view.array B_array = <double[:3*num_tar]> B_c

    B = np.asarray(B_array)
    B.shape = (num_tar, 3)

    return B


def compute_A_and_B_cpp(p, c, Nn, Nb, I_strand, tar, near_field_ratio):
    '''Launch the C++ code to compute the A and B field for given
    conductor bricks, targets and current.

    :param p:
        The nodes of the conductor mesh.

    :param c:
        The the connectivity of the conductor mesh.

    :param Nn:
        The numbers of strands in the normal directions.

    :param Nb:
        The numbers of strands in the bi-normal directions.

    :param I_strand:
        The strand currents.

    :param tar:
        The target positions.

    :param near_field_ratio:
        The near field ratio. If the distance between a segment
        and an evaluation point is smaller than ratio*length,
        where length is the segment length, the interaction
        is computed with all filaments.

    :return:
        The vector potentials and the flux densities in an M x 3 numpy array. 
    '''

    # get the number of points in the mesh
    num_points = p.shape[0]

    # get the number of bricks in the mesh
    num_bricks = c.shape[0]
        
    # get the number of targets
    num_tar = tar.shape[0]

    # ====================================
    # Convert python -> C
    # ====================================

    # make c type data pointers
    cdef np.ndarray[double, ndim=1, mode = 'c'] p_buff = np.ascontiguousarray(p.flatten(), dtype = np.double)
    cdef double* p_c = <double*> p_buff.data

    cdef np.ndarray[int, ndim=1, mode = 'c'] c_buff = np.ascontiguousarray(c.flatten(), dtype = np.int32)
    cdef int* c_c = <int*> c_buff.data
    
    cdef np.ndarray[int, ndim=1, mode = 'c'] Nn_buff = np.ascontiguousarray(Nn.flatten(), dtype = np.int32)
    cdef int* Nn_c = <int*> Nn_buff.data

    cdef np.ndarray[int, ndim=1, mode = 'c'] Nb_buff = np.ascontiguousarray(Nb.flatten(), dtype = np.int32)
    cdef int* Nb_c = <int*> Nb_buff.data

    cdef np.ndarray[double, ndim=1, mode = 'c'] I_str_buff = np.ascontiguousarray(I_strand.flatten(), dtype = np.double)
    cdef double* I_strand_c = <double*> I_str_buff.data

    cdef np.ndarray[double, ndim=1, mode = 'c'] tar_buff = np.ascontiguousarray(tar.flatten(), dtype = np.double)
    cdef double* tar_c = <double*> tar_buff.data

    # ====================================
    # run cpp code
    # ====================================

    cdef double *A_c = <double *> malloc(3*num_tar*sizeof(double))
    cdef double *B_c = <double *> malloc(3*num_tar*sizeof(double))

    ctools.compute_A_and_B(A_c, B_c, p_c, c_c,
                            Nn_c, Nb_c, I_strand_c, tar_c,
                           num_points, num_bricks, num_tar, near_field_ratio)


    # ===========================================
    # Convert C -> python
    # ===========================================
    
    cdef view.array A_array = <double[:3*num_tar]> A_c

    A = np.asarray(A_array)
    A.shape = (num_tar, 3)

    cdef view.array B_array = <double[:3*num_tar]> B_c

    B = np.asarray(B_array)
    B.shape = (num_tar, 3)

    return A, B


def compute_B_cpp(p, c, Nn, Nb, I_strand, tar, near_field_distance):
    '''Launch the C++ code to compute the B field for given
    conductor bricks, targets and currents.

    :param p:
        The nodes of the conductor mesh.

    :param c:
        The the connectivity of the conductor mesh.

    :param Nn:
        The numbers of strands in the normal directions.

    :param Nb:
        The numbers of strands in the bi-normal directions.

    :param I_strand:
        The strand currents.

    :param tar:
        The target positions.

    :param near_field_distance:
        The near field distance.

    :return:
        The vector potentials and the flux densities in an M x 3 numpy array. 
    '''

    # get the number of points in the mesh
    num_points = p.shape[0]

    # get the number of bricks in the mesh
    num_bricks = c.shape[0]
        
    # get the number of targets
    num_tar = tar.shape[0]

    # ====================================
    # Convert python -> C
    # ====================================

    # make c type data pointers
    cdef np.ndarray[double, ndim=1, mode = 'c'] p_buff = np.ascontiguousarray(p.flatten(), dtype = np.double)
    cdef double* p_c = <double*> p_buff.data

    cdef np.ndarray[int, ndim=1, mode = 'c'] c_buff = np.ascontiguousarray(c.flatten(), dtype = np.int32)
    cdef int* c_c = <int*> c_buff.data
    
    cdef np.ndarray[int, ndim=1, mode = 'c'] Nn_buff = np.ascontiguousarray(Nn.flatten(), dtype = np.int32)
    cdef int* Nn_c = <int*> Nn_buff.data

    cdef np.ndarray[int, ndim=1, mode = 'c'] Nb_buff = np.ascontiguousarray(Nb.flatten(), dtype = np.int32)
    cdef int* Nb_c = <int*> Nb_buff.data

    cdef np.ndarray[double, ndim=1, mode = 'c'] I_str_buff = np.ascontiguousarray(I_strand.flatten(), dtype = np.double)
    cdef double* I_strand_c = <double*> I_str_buff.data

    cdef np.ndarray[double, ndim=1, mode = 'c'] tar_buff = np.ascontiguousarray(tar.flatten(), dtype = np.double)
    cdef double* tar_c = <double*> tar_buff.data

    # ====================================
    # run cpp code
    # ====================================

    cdef double *B_c = <double *> malloc(3*num_tar*sizeof(double))

    ctools.compute_B(B_c, p_c, c_c,
                     Nn_c, Nb_c, I_strand_c, tar_c,
                     num_points, num_bricks, num_tar, near_field_distance)


    # ===========================================
    # Convert C -> python
    # ===========================================
    
    cdef view.array B_array = <double[:3*num_tar]> B_c

    B = np.asarray(B_array)
    B.shape = (num_tar, 3)

    return B


def compute_L_cpp(segments, radius, is_open, num_quad_points=6):
    '''Compute the self inductance of a polygon.

    :param segments:
        The segments of the polygon.

    :param radius:
        The polygon radius.

    :param is_open:
        A flag specifying if the polygon is open or closed.

    :param num_quad_points:
        The number of quadrature points for the Gaussian integration.

    :return:
        The self inductance value L. 
    '''

    # get the number of segments
    num_segs = segments.shape[0]

    # the cython interface does not like boolean
    is_open_c = 0

    if is_open:
        is_open_c = 1

    # ====================================
    # Convert python -> C
    # ====================================

    # make c type data pointers
    cdef np.ndarray[double, ndim=1, mode = 'c'] segs_buff = np.ascontiguousarray(segments.flatten(), dtype = np.double)
    cdef double* segs_c = <double*> segs_buff.data

    # ====================================
    # run cpp code
    # ====================================

    L = ctools.compute_L(segs_c, num_segs, radius, num_quad_points, is_open_c)

    return L

def compute_A_iron_cpp(pnt, q, n, w, A, dA):
    '''Compute the magnetic vector potential, due to a magnetized
    iron yoke. This function takes all integration points, normal vectors
    and weights, as well as the vector potentials and normal derivatives
    evaluated at the integration points.

    :param pnt:
        The evaluation points.
    :param q:
        The integration points. 
    :param n:
        The normal vectors.
    :param w:
        The integration weights.
    :param A:
        The vector potetnial at the integration points.
    :param dA:
        The normal derivatives at the integration points.
    :return:
        The magnetic vector potential evaluated there.
    '''

    # get the number of points
    num_pnt = pnt.shape[0]

    # get the number sources
    num_src = q.shape[0]
        
    # ====================================
    # Convert python -> C
    # ====================================

    # make c type data pointers
    cdef np.ndarray[double, ndim=1, mode = 'c'] pnt_buff = np.ascontiguousarray(pnt.flatten(), dtype = np.double)
    cdef double* pnt_c = <double*> pnt_buff.data

    cdef np.ndarray[double, ndim=1, mode = 'c'] q_buff = np.ascontiguousarray(q.flatten(), dtype = np.double)
    cdef double* q_c = <double*> q_buff.data
    
    cdef np.ndarray[double, ndim=1, mode = 'c'] n_buff = np.ascontiguousarray(n.flatten(), dtype = np.double)
    cdef double* n_c = <double*> n_buff.data

    cdef np.ndarray[double, ndim=1, mode = 'c'] w_buff = np.ascontiguousarray(w.flatten(), dtype = np.double)
    cdef double* w_c = <double*> w_buff.data

    cdef np.ndarray[double, ndim=1, mode = 'c'] A_buff = np.ascontiguousarray(A.flatten(), dtype = np.double)
    cdef double* A_c = <double*> A_buff.data

    cdef np.ndarray[double, ndim=1, mode = 'c'] dA_buff = np.ascontiguousarray(dA.flatten(), dtype = np.double)
    cdef double* dA_c = <double*> dA_buff.data

    # ====================================
    # run cpp code
    # ====================================

    cdef double *A_ret_c = <double *> malloc(3*num_pnt*sizeof(double))

    ctools.compute_A_iron(A_ret_c, num_pnt, num_src, pnt_c, q_c, n_c, w_c, A_c, dA_c)


    # ===========================================
    # Convert C -> python
    # ===========================================
    
    cdef view.array A_array = <double[:3*num_pnt]> A_ret_c

    A_ret = np.asarray(A_array)
    A_ret.shape = (num_pnt, 3)

    return A_ret

def compute_B_iron_cpp(pnt, q, n, w, A, dA):
    '''Compute the magnetic flux density, due to a magnetized
    iron yoke. This function takes all integration points, normal vectors
    and weights, as well as the vector potentials and normal derivatives
    evaluated at the integration points.

    :param pnt:
        The evaluation points.
    :param q:
        The integration points. 
    :param n:
        The normal vectors.
    :param w:
        The integration weights.
    :param A:
        The vector potetnial at the integration points.
    :param dA:
        The normal derivatives at the integration points.
    :return:
        The magnetic vector potential evaluated there.
    '''

    # get the number of points
    num_pnt = pnt.shape[0]

    # get the number sources
    num_src = q.shape[0]
        
    # ====================================
    # Convert python -> C
    # ====================================

    # make c type data pointers
    cdef np.ndarray[double, ndim=1, mode = 'c'] pnt_buff = np.ascontiguousarray(pnt.flatten(), dtype = np.double)
    cdef double* pnt_c = <double*> pnt_buff.data

    cdef np.ndarray[double, ndim=1, mode = 'c'] q_buff = np.ascontiguousarray(q.flatten(), dtype = np.double)
    cdef double* q_c = <double*> q_buff.data
    
    cdef np.ndarray[double, ndim=1, mode = 'c'] n_buff = np.ascontiguousarray(n.flatten(), dtype = np.double)
    cdef double* n_c = <double*> n_buff.data

    cdef np.ndarray[double, ndim=1, mode = 'c'] w_buff = np.ascontiguousarray(w.flatten(), dtype = np.double)
    cdef double* w_c = <double*> w_buff.data

    cdef np.ndarray[double, ndim=1, mode = 'c'] A_buff = np.ascontiguousarray(A.flatten(), dtype = np.double)
    cdef double* A_c = <double*> A_buff.data

    cdef np.ndarray[double, ndim=1, mode = 'c'] dA_buff = np.ascontiguousarray(dA.flatten(), dtype = np.double)
    cdef double* dA_c = <double*> dA_buff.data

    # ====================================
    # run cpp code
    # ====================================

    cdef double *B_ret_c = <double *> malloc(3*num_pnt*sizeof(double))

    ctools.compute_B_iron(B_ret_c, num_pnt, num_src, pnt_c, q_c, n_c, w_c, A_c, dA_c)

    # ===========================================
    # Convert C -> python
    # ===========================================
    
    cdef view.array B_array = <double[:3*num_pnt]> B_ret_c

    B_ret = np.asarray(B_array)
    B_ret.shape = (num_pnt, 3)

    return B_ret

def compute_B_eddy_ring(pnt, q, s, w):
    '''Compute the magnetic flux density, due to a spatial distribution of
    Eddy rings. This function takes all integration points, surface current vectors
    and weights evaluated at the integration points, and sums them up rapidly.

    :param pnt:
        The evaluation points.
    :param q:
        The integration points. 
    :param s:
        The normal vectors.
    :param w:
        The integration weights.
    :return:
        The magnetic vector potential evaluated there.
    '''

    # get the number of points
    num_pnt = pnt.shape[0]

    # get the number sources
    num_src = q.shape[0]
    
    # ====================================
    # Convert python -> C
    # ====================================

    # make c type data pointers
    cdef np.ndarray[double, ndim=1, mode = 'c'] pnt_buff = np.ascontiguousarray(pnt.flatten(), dtype = np.double)
    cdef double* pnt_c = <double*> pnt_buff.data

    cdef np.ndarray[double, ndim=1, mode = 'c'] q_buff = np.ascontiguousarray(q.flatten(), dtype = np.double)
    cdef double* q_c = <double*> q_buff.data
    
    cdef np.ndarray[double, ndim=1, mode = 'c'] n_buff = np.ascontiguousarray(s.flatten(), dtype = np.double)
    cdef double* s_c = <double*> n_buff.data

    cdef np.ndarray[double, ndim=1, mode = 'c'] w_buff = np.ascontiguousarray(w.flatten(), dtype = np.double)
    cdef double* w_c = <double*> w_buff.data

    # ====================================
    # run cpp code
    # ====================================

    cdef double *B_ret_c = <double *> malloc(3*num_pnt*sizeof(double))

    ctools.compute_B_eddy_ring(B_ret_c, num_pnt, num_src, pnt_c, q_c, s_c, w_c)

    # ===========================================
    # Convert C -> python
    # ===========================================
    
    cdef view.array B_array = <double[:3*num_pnt]> B_ret_c

    B_ret = np.asarray(B_array)
    B_ret.shape = (num_pnt, 3)

    return B_ret

def compute_B_eddy_ring_mat(pnt, n, nodes, cells, basis, basis_der, q, w):
    '''Compute the dense matrix for the computation of the magnetic flux density,
    due to a spatial distribution of Eddy rings.

    :param pnt:
        The evaluation points.
    :param n:
        The field orientation vectors at the evaluation points.
    :param nodes:
        The nodal coordinates of the mesh.
    :param cells:
        The mesh connectivity matrix.
    :param basis:
        The basis funtions evaluated at the quadrature points.
    :param basis_der:
        The derivatives of the basis functions evaluated at the quadrature points.
    :param q:
        The integration points. 
    :param w:
        The integration weights. 
    :return:
        The evaluation matrix.
    '''

    # get the number of points
    num_pnt = pnt.shape[0]

    # get the number of nodes
    num_nodes = nodes.shape[0] 
    
    # get the number of quadrature points
    num_quad = basis.shape[0]

    # get the number of nodes per element
    num_cell_nodes = basis.shape[1]

    # get the number of cells
    num_cells = np.int64(len(cells)/num_cell_nodes)

    # ====================================
    # Convert python -> C
    # ====================================

    # make c type data pointers
    cdef np.ndarray[double, ndim=1, mode = 'c'] pnt_buff = np.ascontiguousarray(pnt.flatten(), dtype = np.double)
    cdef double* pnt_c = <double*> pnt_buff.data

    cdef np.ndarray[double, ndim=1, mode = 'c'] n_buff = np.ascontiguousarray(n.flatten(), dtype = np.double)
    cdef double* n_c = <double*> n_buff.data

    cdef np.ndarray[double, ndim=1, mode = 'c'] nodes_buff = np.ascontiguousarray(nodes.flatten(), dtype = np.double)
    cdef double* nodes_c = <double*> nodes_buff.data

    cdef np.ndarray[int, ndim=1, mode = 'c'] cell_buff = np.ascontiguousarray(cells.flatten(), dtype = np.int32)
    cdef int* cells_c = <int*> cell_buff.data

    cdef np.ndarray[double, ndim=1, mode = 'c'] basis_buff = np.ascontiguousarray(basis.flatten(), dtype = np.double)
    cdef double* basis_c = <double*> basis_buff.data

    cdef np.ndarray[double, ndim=1, mode = 'c'] basis_der_buff = np.ascontiguousarray(basis_der.flatten(), dtype = np.double)
    cdef double* basis_der_c = <double*> basis_der_buff.data

    cdef np.ndarray[double, ndim=1, mode = 'c'] q_buff = np.ascontiguousarray(q.flatten(), dtype = np.double)
    cdef double* q_c = <double*> q_buff.data

    cdef np.ndarray[double, ndim=1, mode = 'c'] w_buff = np.ascontiguousarray(w.flatten(), dtype = np.double)
    cdef double* w_c = <double*> w_buff.data

    # ====================================
    # run cpp code
    # ====================================

    cdef double *H_mat_c = <double *> malloc(num_nodes*num_pnt*sizeof(double))

    ctools.compute_B_eddy_ring_mat(H_mat_c, num_pnt, pnt_c, n_c, num_nodes, nodes_c, num_cells, cells_c, num_cell_nodes, basis_c, basis_der_c, num_quad, q_c, w_c)

    # ===========================================
    # Convert C -> python
    # ===========================================
    
    cdef view.array H_array = <double[:num_nodes*num_pnt]> H_mat_c

    H = np.asarray(H_array)
    H.shape = (num_pnt, num_nodes)

    return H


def compute_A_mlfmm(src_pts, src_vec, tar_pts, L, max_tree_lvl=3,
                    b_box=[np.zeros((3, )), 0., 1e-6],
                    normals=np.zeros((0, 3)), kind='Biot-Savart'):
    """Compute the magnetic vector potential with the multilevel fast
    multipole method.

    :param src_pts: 
        The source point coordinates.

    :param src_vec: 
        The source vector coordinates.

    :param tar_pts: 
        The target points.
    
    :param L:
        The maximum order of the solid harmonics.

    :param max_tree_level:
        The maximum depth of the cluster tree.

    :param b_box:
        A list [ctr, diam, delta] with the center point of the bounding box and its diameter, as
        well as an additional margin for numerical stability.
        The domain size is
            ctr[0] - 0.5*diam < x < ctr[0] + 0.5*diam
            ctr[1] - 0.5*diam < y < ctr[1] + 0.5*diam
            ctr[2] - 0.5*diam < z < ctr[2] + 0.5*diam

        If diam == 0. the domain size is determined from the data.

    :param normals:
        The normal vectors at the integration points. Required only for some potentials. See below.

    :param kind:
        Specify the kind of potential to evaluate. Options are:
        "Biot-Savart" for conductor fields.
        "Vector-Valued-Single-Layer" for the vector valued single layer potential.
        "Vector-Valued-Double-Layer" for the vector valued double layer potential. (n req)
        For potentials marked with (n req), You need to specify also the normal vectors at the
        integration points.
    """

    # get the number of sources
    num_src = src_pts.shape[0]

    # get the number of targets
    num_tar = tar_pts.shape[0]

    # the Cpp code needs the source and normal vectors in the a common structure we therefore create
    # a (num_src x 6) array with the columns of g and n appended.
    source_array = np.zeros((num_src, 6))
    source_array[:, :3] = src_vec

    # this is the potential specifyer (default Biot-Savart)
    pot_spec = 0

    if kind == 'Vector-Valued-Single-Layer':
        pot_spec = 1

    elif kind == 'Vector-Valued-Double-Layer':
        pot_spec = 2
        if normals.shape[0] == num_src:
            source_array[:, 3:] = normals
        else:
            print('Error: The normal vectors array needs to have the same dimension as the source vector array!')
            return None

    if b_box[1] == 0.:
        min_tar_x = min(tar_pts[:, 0])
        min_tar_y = min(tar_pts[:, 1])
        min_tar_z = min(tar_pts[:, 2])
        max_tar_x = max(tar_pts[:, 0])
        max_tar_y = max(tar_pts[:, 1])
        max_tar_z = max(tar_pts[:, 2])
       
        min_src_x = min(src_pts[:, 0])
        min_src_y = min(src_pts[:, 1])
        min_src_z = min(src_pts[:, 2])
        max_src_x = max(src_pts[:, 0])
        max_src_y = max(src_pts[:, 1])
        max_src_z = max(src_pts[:, 2])

        min_x = min([min_tar_x, min_src_x])
        min_y = min([min_tar_y, min_src_y])
        min_z = min([min_tar_z, min_src_z])

        max_x = max([max_tar_x, max_src_x])
        max_y = max([max_tar_y, max_src_y])
        max_z = max([max_tar_z, max_src_z])

        diam_x = max_x - min_x
        diam_y = max_y - min_y
        diam_z = max_z - min_z

        b_box[0] = np.array([0.5*(min_x + max_x),
                             0.5*(min_y + max_y),
                             0.5*(min_z + max_z)])

        b_box[1] = max([diam_x, diam_y, diam_z])

    # make the domain boundaries
    domain_b = np.zeros((2, 3))
    # some margin for numerical stability
    domain_b[0, 0] = b_box[0][0] - 0.5*b_box[1] - b_box[2]
    domain_b[0, 1] = b_box[0][1] - 0.5*b_box[1] - b_box[2]
    domain_b[0, 2] = b_box[0][2] - 0.5*b_box[1] - b_box[2]

    domain_b[1, 0] = b_box[0][0] + 0.5*b_box[1] + b_box[2]
    domain_b[1, 1] = b_box[0][1] + 0.5*b_box[1] + b_box[2]
    domain_b[1, 2] = b_box[0][2] + 0.5*b_box[1] + b_box[2]

    # ====================================
    # Convert python -> C
    # ====================================

    # make c type data pointers
    cdef np.ndarray[double, ndim=1, mode = 'c'] src_pts_buff = np.ascontiguousarray(src_pts.flatten(), dtype = np.double)
    cdef double* src_pts_c = <double*> src_pts_buff.data

    cdef np.ndarray[double, ndim=1, mode = 'c'] source_array_buff = np.ascontiguousarray(source_array.flatten(), dtype = np.double)
    cdef double* source_array_c = <double*> source_array_buff.data

    cdef np.ndarray[double, ndim=1, mode = 'c'] tar_pts_buff = np.ascontiguousarray(tar_pts.flatten(), dtype = np.double)
    cdef double* tar_pts_c = <double*> tar_pts_buff.data

    cdef np.ndarray[double, ndim=1, mode = 'c'] domain_buff = np.ascontiguousarray(domain_b.flatten(), dtype = np.double)
    cdef double* b_box_c = <double*> domain_buff.data

    # allocate space for the vector potentials
    cdef double *A_c = <double *> malloc(3*num_tar*sizeof(double))

    # ====================================
    # run cpp code
    # ====================================
    
    ctools.compute_A_mlfmm(A_c, num_tar, num_src, src_pts_c, source_array_c, tar_pts_c, L, max_tree_lvl, b_box_c, pot_spec)

    # ===========================================
    # Convert C -> python
    # ===========================================
    
    cdef view.array A_array = <double[:3*num_tar]> A_c

    A = np.asarray(A_array)

    A.shape = (num_tar, 3)

    return A

def compute_B_mlfmm(src_pts, src_vec, tar_pts, L, max_tree_lvl=3, b_box=[np.zeros((3, )), 0., 1e-6],
                    normals=np.zeros((0, 3)), kind='Biot-Savart'):
    """Compute the magnetic flux density with the multilevel fast
    multipole method.

    :param src_pts: 
        The source point coordinates.

    :param src_vec: 
        The source vector coordinates.

    :param tar_pts: 
        The target points.
    
    :param L:
        The maximum order of the solid harmonics.

    :param max_tree_level:
        The maximum depth of the cluster tree.

    :param b_box:
        A list [ctr, diam, delta] with the center point of the bounding box and its diameter, as
        well as an additional margin for numerical stability.
        The domain size is
            ctr[0] - 0.5*diam < x < ctr[0] + 0.5*diam
            ctr[1] - 0.5*diam < y < ctr[1] + 0.5*diam
            ctr[2] - 0.5*diam < z < ctr[2] + 0.5*diam

        If diam == 0. the domain size is determined from the data.

    :param normals:
        The normal vectors at the integration points. Required only for some potentials. See below.

    :param kind:
        Specify the kind of potential to evaluate. Options are:
        "Biot-Savart" for conductor fields.
        "Vector-Valued-Single-Layer" for the vector valued single layer potential.
        "Vector-Valued-Double-Layer" for the vector valued double layer potential. (n req)
        For potentials marked with (n req), You need to specify also the normal vectors at the
        integration points.

    """

    # get the number of sources
    num_src = src_pts.shape[0]

    # get the number of targets
    num_tar = tar_pts.shape[0]

    # the Cpp code needs the source and normal vectors in the a common structure we therefore create
    # a (num_src x 6) array with the columns of g and n appended.
    source_array = np.zeros((num_src, 6))
    source_array[:, :3] = src_vec

    # this is the potential specifyer (default Biot-Savart)
    pot_spec = 0

    if kind == 'Vector-Valued-Single-Layer':
        pot_spec = 1

    elif kind == 'Vector-Valued-Double-Layer':

        pot_spec = 2
        if normals.shape[0] == num_src:
            source_array[:, 3:] = normals
        else:
            print('Error: The normal vectors array needs to have the same dimension as the source vector array!')
            return None

    if b_box[1] == 0.:
        min_tar_x = min(tar_pts[:, 0])
        min_tar_y = min(tar_pts[:, 1])
        min_tar_z = min(tar_pts[:, 2])
        max_tar_x = max(tar_pts[:, 0])
        max_tar_y = max(tar_pts[:, 1])
        max_tar_z = max(tar_pts[:, 2])
       
        min_src_x = min(src_pts[:, 0])
        min_src_y = min(src_pts[:, 1])
        min_src_z = min(src_pts[:, 2])
        max_src_x = max(src_pts[:, 0])
        max_src_y = max(src_pts[:, 1])
        max_src_z = max(src_pts[:, 2])

        min_x = min([min_tar_x, min_src_x])
        min_y = min([min_tar_y, min_src_y])
        min_z = min([min_tar_z, min_src_z])

        max_x = max([max_tar_x, max_src_x])
        max_y = max([max_tar_y, max_src_y])
        max_z = max([max_tar_z, max_src_z])

        diam_x = max_x - min_x
        diam_y = max_y - min_y
        diam_z = max_z - min_z

        b_box[0] = np.array([0.5*(min_x + max_x),
                             0.5*(min_y + max_y),
                             0.5*(min_z + max_z)])

        b_box[1] = max([diam_x, diam_y, diam_z])

    # make the domain boundaries
    domain_b = np.zeros((2, 3))
    # some margin for numerical stability
    domain_b[0, 0] = b_box[0][0] - 0.5*b_box[1] - b_box[2]
    domain_b[0, 1] = b_box[0][1] - 0.5*b_box[1] - b_box[2]
    domain_b[0, 2] = b_box[0][2] - 0.5*b_box[1] - b_box[2]

    domain_b[1, 0] = b_box[0][0] + 0.5*b_box[1] + b_box[2]
    domain_b[1, 1] = b_box[0][1] + 0.5*b_box[1] + b_box[2]
    domain_b[1, 2] = b_box[0][2] + 0.5*b_box[1] + b_box[2]

    # ====================================
    # Convert python -> C
    # ====================================

    # make c type data pointers
    cdef np.ndarray[double, ndim=1, mode = 'c'] src_pts_buff = np.ascontiguousarray(src_pts.flatten(), dtype = np.double)
    cdef double* src_pts_c = <double*> src_pts_buff.data

    cdef np.ndarray[double, ndim=1, mode = 'c'] source_array_buff = np.ascontiguousarray(source_array.flatten(), dtype = np.double)
    cdef double* source_array_c = <double*> source_array_buff.data

    cdef np.ndarray[double, ndim=1, mode = 'c'] tar_pts_buff = np.ascontiguousarray(tar_pts.flatten(), dtype = np.double)
    cdef double* tar_pts_c = <double*> tar_pts_buff.data

    cdef np.ndarray[double, ndim=1, mode = 'c'] domain_buff = np.ascontiguousarray(domain_b.flatten(), dtype = np.double)
    cdef double* b_box_c = <double*> domain_buff.data

    # allocate space for the vector potentials
    cdef double *B_c = <double *> malloc(3*num_tar*sizeof(double))

    # ====================================
    # run cpp code
    # ====================================
    
    ctools.compute_B_mlfmm(B_c, num_tar, num_src, src_pts_c, source_array_c, tar_pts_c, L, max_tree_lvl, b_box_c, pot_spec)

    # ===========================================
    # Convert C -> python
    # ===========================================
    
    cdef view.array B_array = <double[:3*num_tar]> B_c

    B = np.asarray(B_array)

    B.shape = (num_tar, 3)

    return B

def compute_B_roxie_mlfmm(src_pts_coil, src_vec_coil, src_pts_iron,
                          src_vec_A, src_vec_dA, normals, tar_pts, L,
                          max_tree_lvl=3, b_box=[np.zeros((3, )), 0., 1e-6]):
    """Compute the magnetic flux density with the multilevel fast
    multipole method for the sources (coil and iron) coming from a ROXIE
    simulation.

    :param src_pts_coil: 
        The source point coordinates for the coils.

    :param src_vec_coil: 
        The source vector coordinates for the coils.

    :param src_pts_iron: 
        The source point coordinates for the iron.

    :param src_vec_A: 
        The source vector coordinates for the vector potential.

    :param src_vec_dA: 
        The source vector coordinates for the vector potential normal derivative.

    :param normals: 
        The the normal vectors on src_pts_iron.

    :param tar_pts: 
        The target points.
    
    :param L:
        The maximum order of the solid harmonics.

    :param max_tree_level:
        The maximum depth of the cluster tree.

    :param b_box:
        A list [ctr, diam, delta] with the center point of the bounding box and its diameter, as
        well as an additional margin for numerical stability.
        The domain size is
            ctr[0] - 0.5*diam < x < ctr[0] + 0.5*diam
            ctr[1] - 0.5*diam < y < ctr[1] + 0.5*diam
            ctr[2] - 0.5*diam < z < ctr[2] + 0.5*diam

        If diam == 0. the domain size is determined from the data.

    """

    # get the number of sources
    num_src_coil = src_pts_coil.shape[0]
    num_src_iron = src_pts_iron.shape[0]

    # get the number of targets
    num_tar = tar_pts.shape[0]

    # the Cpp code needs the source and normal vectors in the a common structure we therefore create
    # a (num_src x 6) array with the columns of g and n appended.
    source_coil_array = np.zeros((num_src_coil, 6))
    source_coil_array[:, :3] = src_vec_coil

    source_A_array = np.zeros((num_src_iron, 6))
    source_A_array[:, :3] = src_vec_A
    source_A_array[:, 3:] = normals

    source_dA_array = np.zeros((num_src_iron, 6))
    source_dA_array[:, :3] = src_vec_dA

    # this is for the bounding box. Maybe write a function in the future.
    if b_box[1] == 0.:
        min_tar_x = min(tar_pts[:, 0])
        min_tar_y = min(tar_pts[:, 1])
        min_tar_z = min(tar_pts[:, 2])
        max_tar_x = max(tar_pts[:, 0])
        max_tar_y = max(tar_pts[:, 1])
        max_tar_z = max(tar_pts[:, 2])
       
        min_src_coil_x = min(src_pts_coil[:, 0])
        min_src_coil_y = min(src_pts_coil[:, 1])
        min_src_coil_z = min(src_pts_coil[:, 2])
        max_src_coil_x = max(src_pts_coil[:, 0])
        max_src_coil_y = max(src_pts_coil[:, 1])
        max_src_coil_z = max(src_pts_coil[:, 2])

        min_src_iron_x = min(src_pts_iron[:, 0])
        min_src_iron_y = min(src_pts_iron[:, 1])
        min_src_iron_z = min(src_pts_iron[:, 2])
        max_src_iron_x = max(src_pts_iron[:, 0])
        max_src_iron_y = max(src_pts_iron[:, 1])
        max_src_iron_z = max(src_pts_iron[:, 2])

        min_x = min([min_tar_x, min_src_iron_x, min_src_coil_x])
        min_y = min([min_tar_y, min_src_iron_y, min_src_coil_y])
        min_z = min([min_tar_z, min_src_iron_z, min_src_coil_z])

        max_x = max([max_tar_x, max_src_iron_x, max_src_coil_x])
        max_y = max([max_tar_y, max_src_iron_y, max_src_coil_y])
        max_z = max([max_tar_z, max_src_iron_z, max_src_coil_z])

        diam_x = max_x - min_x
        diam_y = max_y - min_y
        diam_z = max_z - min_z

        b_box[0] = np.array([0.5*(min_x + max_x),
                             0.5*(min_y + max_y),
                             0.5*(min_z + max_z)])

        b_box[1] = max([diam_x, diam_y, diam_z])

    # make the domain boundaries
    domain_b = np.zeros((2, 3))
    # some margin for numerical stability
    domain_b[0, 0] = b_box[0][0] - 0.5*b_box[1] - b_box[2]
    domain_b[0, 1] = b_box[0][1] - 0.5*b_box[1] - b_box[2]
    domain_b[0, 2] = b_box[0][2] - 0.5*b_box[1] - b_box[2]

    domain_b[1, 0] = b_box[0][0] + 0.5*b_box[1] + b_box[2]
    domain_b[1, 1] = b_box[0][1] + 0.5*b_box[1] + b_box[2]
    domain_b[1, 2] = b_box[0][2] + 0.5*b_box[1] + b_box[2]

    # ====================================
    # Convert python -> C
    # ====================================

    # make c type data pointers
    cdef np.ndarray[double, ndim=1, mode = 'c'] src_pts_coil_buff = np.ascontiguousarray(src_pts_coil.flatten(), dtype = np.double)
    cdef double* src_pts_coil_c = <double*> src_pts_coil_buff.data

    cdef np.ndarray[double, ndim=1, mode = 'c'] source_coil_array_buff = np.ascontiguousarray(source_coil_array.flatten(), dtype = np.double)
    cdef double* source_coil_array_c = <double*> source_coil_array_buff.data

    cdef np.ndarray[double, ndim=1, mode = 'c'] src_pts_iron_buff = np.ascontiguousarray(src_pts_iron.flatten(), dtype = np.double)
    cdef double* src_pts_iron_c = <double*> src_pts_iron_buff.data

    cdef np.ndarray[double, ndim=1, mode = 'c'] source_A_buff = np.ascontiguousarray(source_A_array.flatten(), dtype = np.double)
    cdef double* source_A_c = <double*> source_A_buff.data

    cdef np.ndarray[double, ndim=1, mode = 'c'] source_dA_buff = np.ascontiguousarray(source_dA_array.flatten(), dtype = np.double)
    cdef double* source_dA_c = <double*> source_dA_buff.data

    cdef np.ndarray[double, ndim=1, mode = 'c'] tar_pts_buff = np.ascontiguousarray(tar_pts.flatten(), dtype = np.double)
    cdef double* tar_pts_c = <double*> tar_pts_buff.data

    cdef np.ndarray[double, ndim=1, mode = 'c'] domain_buff = np.ascontiguousarray(domain_b.flatten(), dtype = np.double)
    cdef double* b_box_c = <double*> domain_buff.data

    # allocate space for the vector potentials
    cdef double *B_c = <double *> malloc(3*num_tar*sizeof(double))

    # ====================================
    # run cpp code
    # ====================================
    
    ctools.compute_B_roxie_mlfmm(B_c, num_tar, num_src_coil, src_pts_coil_c,
                                 source_coil_array_c, num_src_iron, src_pts_iron_c,
                                 source_A_c,
                                 source_dA_c, tar_pts_c, L, max_tree_lvl, b_box_c)

    # ===========================================
    # Convert C -> python
    # ===========================================
    
    cdef view.array B_array = <double[:3*num_tar]> B_c

    B = np.asarray(B_array)

    B.shape = (num_tar, 3)

    return B

def compute_solid_harmonics_cpp(p, c, Nn, Nb, I_strand, tar, L, r_ref, num_quad_points, near_field_distance):
    '''Launch the C++ code to compute the solid harmonics for given
    conductor bricks, targets and currents.

    :param p:
        The nodes of the conductor mesh.

    :param c:
        The the connectivity of the conductor mesh.

    :param Nn:
        The numbers of strands in the normal directions.

    :param Nb:
        The numbers of strands in the bi-normal directions.

    :param I_strand:
        The strand currents.

    :param tar:
        The target positions.

    :param L:
        The maximum degree of the solid harmonics.

    :param r_ref:
        A reference radius.

    :param num_quad_points:
        Number of quadrature points.

    :param near_field_distance:
        The near field distance.

    :return:
        The vector potentials and the flux densities in an M x 3 numpy array. 
    '''

    # get the number of points in the mesh
    num_points = p.shape[0]

    # get the number of bricks in the mesh
    num_bricks = c.shape[0]
        
    # get the number of targets
    num_tar = tar.shape[0]

    # the number of solid harmonic coefficients
    num_coeffs = (L + 1)**2

    # ====================================
    # Convert python -> C
    # ====================================

    # make c type data pointers
    cdef np.ndarray[double, ndim=1, mode = 'c'] p_buff = np.ascontiguousarray(p.flatten(), dtype = np.double)
    cdef double* p_c = <double*> p_buff.data

    cdef np.ndarray[int, ndim=1, mode = 'c'] c_buff = np.ascontiguousarray(c.flatten(), dtype = np.int32)
    cdef int* c_c = <int*> c_buff.data
    
    cdef np.ndarray[int, ndim=1, mode = 'c'] Nn_buff = np.ascontiguousarray(Nn.flatten(), dtype = np.int32)
    cdef int* Nn_c = <int*> Nn_buff.data

    cdef np.ndarray[int, ndim=1, mode = 'c'] Nb_buff = np.ascontiguousarray(Nb.flatten(), dtype = np.int32)
    cdef int* Nb_c = <int*> Nb_buff.data

    cdef np.ndarray[double, ndim=1, mode = 'c'] I_str_buff = np.ascontiguousarray(I_strand.flatten(), dtype = np.double)
    cdef double* I_strand_c = <double*> I_str_buff.data

    cdef np.ndarray[double, ndim=1, mode = 'c'] tar_buff = np.ascontiguousarray(tar.flatten(), dtype = np.double)
    cdef double* tar_c = <double*> tar_buff.data

    # ====================================
    # run cpp code
    # ====================================

    cdef double *M_real_c = <double *> malloc(3*num_tar*num_coeffs*sizeof(double))
    cdef double *M_imag_c = <double *> malloc(3*num_tar*num_coeffs*sizeof(double))

    ctools.compute_solid_harmonics(M_real_c,
                            M_imag_c,
                            p_c,
                            c_c,
                            Nn_c,
                            Nb_c,
                            I_strand_c,
                            tar_c,
                            num_points,
                            num_bricks,
                            num_tar,
                            L,
                            r_ref,
                            num_quad_points,
                            near_field_distance)

    # ===========================================
    # Convert C -> python
    # ===========================================
    
    cdef view.array M_real_array = <double[:3*num_tar*num_coeffs]> M_real_c
    cdef view.array M_imag_array = <double[:3*num_tar*num_coeffs]> M_imag_c

    M_real = np.asarray(M_real_array)
    M_imag = np.asarray(M_imag_array)

    M_real.shape = (3*num_tar, num_coeffs)
    M_imag.shape = (3*num_tar, num_coeffs)

    return M_real + 1j*M_imag