#ifndef EVALUATORS_H
#define EVALUATORS_H

#include <Eigen/Core>

#include "convertors.h"
#include "mesh_tools.h"
#include "analytical.h"

// =======================================
// Test the cpp tools to evaluate fields
// fast
// 
// Author: Melvin Liebsch
// email: melvin.liebsch@cern.ch
// =======================================


/**
    * Evaluate the B field based on the (5.51) in Field Computation for Accelerator Magnets.
    * by S. Russenschuck.
    * Here we consider a collection of line segments as sources.
    * 
    * @param ret_data The return data pointer. You need to have allocated it first to the
                      correct size.
    * @param src The source points in a c vector.
    * @param tar The target points in a c vector.
    * @param current The magnet current.
    * @param num_src The number of sources.
    * @param num_tar The number of targets.
    * @return Nothing.
*/
void compute_B_line_segs(double* ret_data, const double *src_ptr, const double *tar_ptr, const  double current, const int num_src, const int num_tar);


/**
    * Evaluate the vector potential and the B field based on the equations (5.50) and (5.51) 
    * in "Field Computation for Accelerator Magnets" by S. Russenschuck
    * This function is based on 8 noded brick elements for the conductor geometry.
    * We approximate far field interactions by single line elements, while near
    * field interactions are calculated with a stranded conductor model.
    * 
    * @param ret_data_A The return data pointer for the magnetic vector potential.
                        You need to have allocated it first to the correct size.
    * @param ret_data_B The return data pointer for the magnetic flux density.
                        You need to have allocated it first to the correct size.
    * @param p The nodal coordinates of the conductor mesh.
    * @param c The connectivity of the conductor mesh.
    * @param tar The target points in a c vector.
    * @param N_n The number of strands along the normal vector N.
    * @param N_b The number of strands along the bi-normal vector B.
    * @param I_strand The strand currents.
    * @param I_strand The strand currents.
    * @param tar_ptr The coordinates of the target points.
    * @param num_points The number of nodes in the mesh.
    * @param num_bricks The number of brick elements.
    * @param num_tar The number of targets.
    * @param near_field_distance The distance after which interactions are
                                 considered far field.
    * @return Nothing.
*/
void compute_A_and_B(double* ret_data_A,
                     double* ret_data_B,
                     const double *p,
                     const int *c,
                     const int *N_n,
                     const int *N_b,
                     const double *I_strand,
                     const double *tar_ptr,
                     const int num_points,
                     const int num_bricks,
                     const int num_tar,
                     const double near_field_distance);

/**
    * Evaluate the B field based on the equations (5.50) and (5.51) 
    * in "Field Computation for Accelerator Magnets" by S. Russenschuck
    * This function is based on 8 noded brick elements for the conductor geometry.
    * We approximate far field interactions by single line elements, while near
    * field interactions are calculated with a stranded conductor model.
    * 
    * @param ret_data_B The return data pointer for the magnetic flux density.
                        You need to have allocated it first to the correct size.
    * @param p The nodal coordinates of the conductor mesh.
    * @param c The connectivity of the conductor mesh.
    * @param tar The target points in a c vector.
    * @param N_n The number of strands along the normal vector N.
    * @param N_b The number of strands along the bi-normal vector B.
    * @param I_strand The strand currents.
    * @param I_strand The strand currents.
    * @param tar_ptr The coordinates of the target points.
    * @param num_points The number of nodes in the mesh.
    * @param num_bricks The number of brick elements.
    * @param num_tar The number of targets.
    * @param near_field_distance The distance after which interactions are
                                 considered far field.
    * @return Nothing.
*/
void compute_B(double* ret_data_B,
                     const double *p,
                     const int *c,
                     const int *N_n,
                     const int *N_b,
                     const double *I_strand,
                     const double *tar_ptr,
                     const int num_points,
                     const int num_bricks,
                     const int num_tar,
                     const double near_field_distance);

/**
    * Compute the self inductance of a segmented polygonal conductor.
    * 
    * @param segments The segments in a C style array.
    * @param num_segs The number of segments.
    * @param radius The radius of the conductor.
    * @param num_points The number of points for the Gaussian integration.
    * @param is_open An integer specifying if the conductor is open. 1 means yes.
    * @return The inductance.
*/
double compute_L(const double *segs, const int num_segs, const double radius, const int num_points, const int is_open);

/**
    * Compute the magnetic vector potential, due to a magnetized
    * iron yoke. This function takes all integration points, normal vectors
    * and weights, as well as the vector potentials and normal derivatives
    * evaluated at the integration points.
    *  
    * @param ret_A The magnetic vector potential.
    * @param num_pnt The number of points.
    * @param num_src The number of sources.
    * @param pnt The evaluation points.
    * @param q The integration points. 
    * @param n The normal vectors.
    * @param w The integration weights.
    * @param A The vector potetnial at the integration points.
    * @param dA The normal derivatives at the integration points.
    * @return Nothing.
*/
void compute_A_iron(double *ret_A,
                    const int num_pnt,
                    const int num_src,
                    const double *pnt,
                    const double *q,
                    const double *n,
                    const double *w,
                    const double *A, 
                    const double *dA);

/**
    * Compute the magnetic flux density, due to a magnetized
    * iron yoke. This function takes all integration points, normal vectors
    * and weights, as well as the vector potentials and normal derivatives
    * evaluated at the integration points.
    *  
    * @param ret_B The magnetic vector potential.
    * @param num_pnt The number of points.
    * @param num_src The number of sources.
    * @param pnt The evaluation points.
    * @param q The integration points. 
    * @param n The normal vectors.
    * @param w The integration weights.
    * @param A The vector potetnial at the integration points.
    * @param dA The normal derivatives at the integration points.
    * @return Nothing.
*/
void compute_B_iron(double *ret_B,
                    const int num_pnt,
                    const int num_src,
                    const double *pnt,
                    const double *q,
                    const double *n,
                    const double *w,
                    const double *A, 
                    const double *dA);

/**
    * Compute the magnetic flux density, due to a spatial distribution of
    * Eddy rings. This function takes all integration points, surface current vectors
    * and weights evaluated at the integration points, and sums them up rapidly.
    *  
    * @param ret_B The magnetic vector potential.
    * @param num_pnt The number of points.
    * @param num_src The number of sources.
    * @param pnt The evaluation points.
    * @param q The integration points. 
    * @param s The surface current vectors.
    * @param w The integration weights.
    * @return Nothing.
*/
void compute_B_eddy_ring(double *ret_B,
                    const int num_pnt,
                    const int num_src,
                    const double *pnt,
                    const double *q,
                    const double *s,
                    const double *w);

/**
    * Compute the matrix for the evaluation of the magnetic flux density components, 
    * due to a spatial distribution of Eddy rings. 
    *  
    * @param ret_mat The return matrix,
    * @param num_pnt The number of target points.
    * @param pnt The target points.
    * @param n The orientation vectors at the target points.
    * @param num_nodes The number of nodes.
    * @param nodes The nodal coordinates.
    * @param num_cells The number of cells.
    * @param cells The cells.
    * @param num_cell_nodes The number of nodes per cell.
    * @param basis The basis functions evaluated at the quadrature points.
    * @param basis_der The derivatives of the basis functions evaluated at the
                        quadrature points.
    * @param num_quad The number of quadrature points. 
    * @param q The integration points. 
    * @param w The integration weights.
    * @return Nothing.
*/
void compute_B_eddy_ring_mat(double *ret_mat,
                    const int num_pnt,
                    const double *pnt,
                    const double *n,
                    const int num_nodes,
                    const double *nodes,
                    const int num_cells,
                    const int *cells,
                    const int num_cell_nodes,
                    const double *basis,
                    const double *basis_der,
                    const int num_quad,
                    const double *q,
                    const double *w);

/**
    * Compute the magnetic vector potential with the multilevel fast
    * multipole method.
    *  
    * @param A A pointer to the return data,
    * @param num_tar The number of target points.
    * @param num_src The number of source points.
    * @param src_pts The source points pointer.
    * @param src_vec The source vectors pointer.
    * @param tar_pts The target points pointer.
    * @param L The maximum order of the solid harmonics.
    * @param max_tree_lvl The maximum depth of the cluster tree.
    * @param b_box_c The bounding box of the domain covering all sources and targets.
    * @param potential_spec An integer specifying the potential kind. Options are: 
                            (0) Magnetic vector potential (Biot-Savart)
                            (1) Vector-Valued-Single-Layer (For iron magnetization dA)
                            (2) Vector-Valued-Double-Layer (For iron magnetization A)
    * @return Nothing.
*/
void compute_A_mlfmm(double *A,
                    const int num_tar,
                    const int num_src,
                    const double *src_pts,
                    const double *src_vec,
                    const double *tar_pts,
                    const int L,
                    const int max_tree_lvl,
                    const double *b_box_c,
                    const int potential_spec);

/**
    * Compute the flux density with the multilevel fast
    * multipole method.
    *  
    * @param B A pointer to the return data,
    * @param num_tar The number of target points.
    * @param num_src The number of source points.
    * @param src_pts The source points pointer.
    * @param src_vec The source vectors pointer.
    * @param tar_pts The target points pointer.
    * @param L The maximum order of the solid harmonics.
    * @param max_tree_lvl The maximum depth of the cluster tree.
    * @param b_box_c The bounding box of the domain covering all sources and targets.
    * @return Nothing.
*/
void compute_B_mlfmm(double *B,
                    const int num_tar,
                    const int num_src,
                    const double *src_pts,
                    const double *src_vec,
                    const double *tar_pts,
                    const int L,
                    const int max_tree_lvl,
                    const double *b_box_c,
                    const int potential_spec);

/**
    * Compute the flux density with the multilevel fast
    * multipole method for the result of a magnet simulation in ROXIE.
    * We compute all farfield interactions together to save time.
    *  
    * @param B A pointer to the return data,
    * @param num_tar The number of target points.
    * @param num_src_coil The number of source points for the coil.
    * @param src_pts_coil The source points pointer for the coil.
    * @param src_vec_coil The source vectors pointer for the coil.
    * @param num_src_iron The number of source points for the vector potential at the iron air interface.
    * @param src_pts_iron The source points pointer for the vector potential at the iron air interface.
    * @param src_vec_A The source vectors pointer for the vector potential at the iron air interface.
    * @param src_vec_dA The source vectors pointer for the normal derivative of the vector potential at the iron air interface.
    * @param tar_pts The target points pointer.
    * @param L The maximum order of the solid harmonics.
    * @param max_tree_lvl The maximum depth of the cluster tree.
    * @param b_box_c The bounding box of the domain covering all sources and targets.
    * @param potential_spec An integer specifying the potential kind. Options are: 
                          (0) Curl of Magnetic vector potential (Biot-Savart)
                          (1) Curl of Vector-Valued-Single-Layer (For iron magnetization dA)
                          (2) Curl of Vector-Valued-Double-Layer (For iron magnetization A)
    * @return Nothing.
*/
void compute_B_roxie_mlfmm(double *B,
                    const int num_tar,
                    const int num_src_coil,
                    const double *src_pts_coil,
                    const double *src_vec_coil,
                    const int num_src_iron,
                    const double *src_pts_iron,
                    const double *src_vec_A,
                    const double *src_vec_dA,
                    const double *tar_pts,
                    const int L,
                    const int max_tree_lvl,
                    const double *b_box_c);
                    
/**
    * Compute the solid harmonics up to degree L at given positions.
    *  
    * @param ret_real The real parts of the coefficients.
    * @param ret_imag The imaginary parts of the coefficient.
    * @param p The nodal coordinates of the conductor mesh.
    * @param c The connectivity of the conductor mesh.
    * @param tar The target points in a c vector.
    * @param N_n The number of strands along the normal vector N.
    * @param N_b The number of strands along the bi-normal vector B.
    * @param I_strand The strand currents.
    * @param tar_ptr The coordinates of the target points.
    * @param num_points The number of nodes in the mesh.
    * @param num_bricks The number of brick elements.
    * @param num_tar The number of targets.
    * @param L The maximum degree of the solid harmonics.
    * @param r_ref A reference radius.
    * @param num_quad_points Number of quadrature points.
    * @param near_field_distance The distance after which interactions are
                                 considered far field.
    * @return Nothing.
*/
void compute_solid_harmonics(double *ret_real,
                     double *ret_imag,
                     const double *p,
                     const int *c,
                     const int *N_n,
                     const int *N_b,
                     const double *I_strand,
                     const double *tar_ptr,
                     const int num_points,
                     const int num_bricks,
                     const int num_tar,
                     const int L,
                     const int r_ref,
                     const int num_quad_points,
                     const double near_field_distance);

#endif
