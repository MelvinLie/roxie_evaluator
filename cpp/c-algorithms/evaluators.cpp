#include <Eigen/Dense>
#include <Eigen/Core>
#include <iostream>
#include <chrono>

#include "convertors.h"
#include "mesh_tools.h"
#include "analytical.h"
#include "inductance_calculation.h"
#include "solid_harmonics.hpp"
#include "boundary_elements.h"
#include "MLFMM.hpp"
#include "MagneticVectorPotentialMonitor.hpp"
#include "MagneticFluxDensityMonitor.hpp"
#include "VectorSingleLayerMonitor.hpp"
#include "VectorDoubleLayerMonitor.hpp"
#include "CurlVectorSingleLayerMonitor.hpp"
#include "CurlVectorDoubleLayerMonitor.hpp"

#define PI 3.14159265358979323846264338327950

void compute_B_line_segs(double* ret_data, const double *src_ptr, const double *tar_ptr, const double current, const int num_src, const int num_tar){
/**
    * Evaluate the B field based on the (5.51) in Field Computation for Accelerator Magnets.
    * by S. Russenschuck
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

    //  build the Eigen style matrices.
    //  for this simple code, it might be overkill to go
    //  to Eigen... but the code was originally developed in Eigen Cpp,
    //  and I was yet too lazy to rewrite it.
    Eigen::MatrixXd src = build_MatrixXd(src_ptr, num_src, 6);
    Eigen::MatrixXd tar = build_MatrixXd(tar_ptr, num_tar, 3);
    
    // for time measurement
    //  std::chrono::time_point<std::chrono::steady_clock> t_start, t_end;
    //  std::chrono::duration<double> t_el;

    // flag to set if output is desired
    bool print = false;

    if(num_tar > 3000){
        // print out the details
        std::cout << "********************************" << std::endl;
        std::cout << "  CCT evaluator  " << std::endl;
        std::cout << "********************************" << std::endl;
        std::cout << "number of sources = " << num_src << std::endl;
        std::cout << "number of field points = " << num_tar << std::endl;
        print = true;
    }
  
    // make space for results
    Eigen::MatrixXd B(num_tar, 3);
    B.setZero();

    if (print) std::cout << "start computation" << std::endl;

    //  t_start =  std::chrono::steady_clock::now();

    #pragma omp parallel
    {
        // make space for results
        Eigen::MatrixXd my_B(num_tar, 3);
        my_B.setZero();

        // make space for difference vector
        Eigen::Vector3d d1, d2;

        // norms
        double n_d1, n_d2;

        // scalar products
        double d1_d2;

        // factor applied to all 3 components
        double factor;

        for (int i = 0; i < num_tar ; ++i){

        #pragma omp single nowait
        {
            for(int j = 0; j < num_src ; ++j){

                //  difference vectors
                d1 = (src.row(j).segment(0,3) - tar.row(i)).transpose();
                d2 = (src.row(j).segment(3,3) - tar.row(i)).transpose();


                //  norms
                n_d1 = d1.norm();
                n_d2 = d2.norm();

                //  scalar product
                d1_d2 = (d1.array() * d2.array()).sum();
                
                //  factor
                factor = (n_d1 + n_d2)/(n_d1*n_d2 + d1_d2)/n_d1/n_d2;

                //  Bx ~ (d1y d2z) - (d1z d2y)
                my_B(i,0) += (d1(1)*d2(2) - d1(2)*d2(1))*factor;

                //  By ~ (d1z d2x) - (d1x d2z)
                my_B(i,1) += (d1(2)*d2(0) - d1(0)*d2(2))*factor;

                //  Bz ~ (d1x d2y) - (d1y d2x)
                my_B(i,2) += (d1(0)*d2(1) - d1(1)*d2(0))*factor;


            }
        }
        }
        #pragma omp critical
        {
            B += my_B;
            
        }
    }

    B *= current*1e-7;

    //  t_end =  std::chrono::steady_clock::now();
    //  t_el = t_end - t_start;

    //  if (print) std::cout << "elapsed time = " << t_el.count() << " sec." << std::endl;
    
    //  we now copy the data to the c_style pointer
    //  ML: The direct transfer from an Eigen::MatrixXd is quite
    //  elaborate. I want to keep it easily, so I just copy.
    //  Maybe in the future we can improve this.
    to_c_array(B, ret_data);

    return;

}

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
                     const double near_field_ratio){
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
        * @param near_field_ratio The near field ratio. If the distance between a segment
                                  and an evaluation point is smaller than ratio*length,
                                  where length is the segment length, the interaction
                                  is computed with all filaments.
        * @return Nothing.
    */


  //  for time measurement
  std::chrono::time_point<std::chrono::steady_clock> t_start, t_end;
  std::chrono::duration<double> t_el;

  //  make eigen objects
  Eigen::MatrixXd c_points = build_MatrixXd(p, num_points, 3);
  Eigen::VectorXd I_str = build_VectorXd(I_strand, num_bricks);
  Eigen::MatrixXi c_bricks = build_MatrixXi(c, num_bricks, 8);
  Eigen::MatrixXd pnt = build_MatrixXd(tar_ptr, num_tar, 3);

  // flag to set if output is desired
  bool print = false;

  if(num_tar > 3000){
    // print out the details
    std::cout << "********************************" << std::endl;
    std::cout << "  field evaluator  " << std::endl;
    std::cout << "********************************" << std::endl;
    std::cout << "number of conductor segments = " << num_bricks << std::endl;
    std::cout << "number of field points = " << num_tar << std::endl;
    // std::cout << "number of interactions = " << num_tar*num_src << std::endl;  // this most certaintly overflows
    print = true;
  }
  
  // make space for results
  Eigen::MatrixXd B(num_tar,3);  // flux density
  Eigen::MatrixXd A(num_tar,3);  // magnetic vector potential
  B.setZero();
  A.setZero();


  if (print) std::cout << "start computation" << std::endl;

  t_start =  std::chrono::steady_clock::now();

  #pragma omp parallel
  {
    // make space for results
    Eigen::MatrixXd my_A(num_tar,3);
    Eigen::MatrixXd my_B(num_tar,3);
    my_A.setZero();
    my_B.setZero();

    // make space for auxilliary vectors
    Eigen::Vector3d r1, r2, c, d, vel;

    // this is a container for the difference vectors,
    //  i.e. the brick corners minus the observation point
    Eigen::MatrixXd r_points = c_points;

    // norms
    double dist;
    double length;

    // coordinates
    double u, v;


    for (int i = 0; i < num_tar ; ++i){


      // subtract the ovservation points
      r_points.col(0) = c_points.col(0).array() - pnt(i,0);
      r_points.col(1) = c_points.col(1).array() - pnt(i,1);
      r_points.col(2) = c_points.col(2).array() - pnt(i,2);

      #pragma omp single nowait
      {
      for(int j = 0; j < num_bricks ; ++j){
        
        // the line starts here
        r1 = eval_polygon(0.5, 0.5, r_points.row(c_bricks(j, 0)),
                                        r_points.row(c_bricks(j, 1)),
                                        r_points.row(c_bricks(j, 2)),
                                        r_points.row(c_bricks(j, 3)));

        // the line ends here
        r2 = eval_polygon(0.5, 0.5, r_points.row(c_bricks(j, 4)),
                                        r_points.row(c_bricks(j, 5)),
                                        r_points.row(c_bricks(j, 6)),
                                        r_points.row(c_bricks(j, 7)));


        // get the midpoint
        d = -0.5*(r2 + r1);

        // get the length of this segment
        length = (r2 - r1).norm();

        // distance
        dist = d.norm();

        // check for near field
        if (dist < near_field_ratio*length){
          
          // n direction
          for (int l = 0; l < N_n[j]; ++l){
            
            // the strand u coordinate
            u = (2*l+1.)/(2*N_n[j]);


            // b direction
            for (int m = 0; m < N_b[j]; ++m){

              // the strand v coordinate
              v = (2*m+1.)/(2*N_b[j]);

              // the line starts here
              r1 = eval_polygon(u, v, r_points.row(c_bricks(j,0)),
                                              r_points.row(c_bricks(j,1)),
                                              r_points.row(c_bricks(j,2)),
                                              r_points.row(c_bricks(j,3)));


              // the line ends here
              r2 = eval_polygon(u, v, r_points.row(c_bricks(j,4)),
                                              r_points.row(c_bricks(j,5)),
                                              r_points.row(c_bricks(j,6)),
                                              r_points.row(c_bricks(j,7)));

              compute_integrals_current_segment(r1, r2, &my_A, &my_B, I_str(j), i);
            }


          }

        }

        else{
          
          compute_integrals_current_segment(r1, r2, &my_A, &my_B, I_str(j)*N_n[j]*N_b[j], i);

        }

      }
    }
    }
    
    #pragma omp critical
    {

      A += my_A;
      B += my_B;
    }
  }

  // apply factor mu_0/(4 pi)
  A *= 1e-7;
  B *= 1e-7;
  
  t_end =  std::chrono::steady_clock::now();
  t_el = t_end - t_start;

  if (print) std::cout << "elapsed time = " << t_el.count() << " sec." << std::endl;

  //  copy data to return pointers
  to_c_array(A, ret_data_A);
  to_c_array(B, ret_data_B);

  return;


}

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
                     const double near_field_distance){

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
      * @param tar_ptr The coordinates of the target points.
      * @param num_points The number of nodes in the mesh.
      * @param num_bricks The number of brick elements.
      * @param num_tar The number of targets.
      * @param near_field_distance The distance after which interactions are
                                  considered far field.
      * @return Nothing.
  */

  // See the technical documentation for the definition of the input files

  //  make eigen objects
  Eigen::MatrixXd c_points = build_MatrixXd(p, num_points, 3);
  Eigen::VectorXd I_str = build_VectorXd(I_strand, num_bricks);
  Eigen::MatrixXi c_bricks = build_MatrixXi(c, num_bricks, 8);
  Eigen::MatrixXd pnt = build_MatrixXd(tar_ptr, num_tar, 3);


  // flag to set if output is desired
  bool print = false;

  if(num_tar > 1){
    // print out the details
    std::cout << "********************************" << std::endl;
    std::cout << "  flux density evaluator  " << std::endl;
    std::cout << "********************************" << std::endl;
    std::cout << "number of conductor bricks = " << num_bricks << std::endl;
    std::cout << "number of nodes = " << num_points << std::endl;
    std::cout << "number of field points = " << num_tar << std::endl;
    // std::cout << "number of interactions = " << num_points*num_src << std::endl;  // this most certaintly overflows
    print = true;
  }


  // make space for results
  Eigen::MatrixXd B(num_tar, 3);  // flux density
  B.setZero();


  #pragma omp parallel
  {
    // make space for results
    Eigen::MatrixXd my_B(num_tar,3);
    my_B.setZero();

    // make space for auxilliary vectors
    Eigen::Vector3d r1, r2, c, d, vel;

    // this is a container for the difference vectors,
    //  i.e. the brick corners minus the observation point
    Eigen::MatrixXd r_points = c_points;

    // norms
    double dist;

    // coordinates
    double u,v;

    for (int i = 0; i < num_tar ; ++i){


      #pragma omp single nowait
      {
      for(int j = 0; j < num_bricks ; ++j){

        // the line starts here (relative to the observation point)
        r1 = eval_polygon(0.5, 0.5, r_points.row(c_bricks(j,0)),
                                        r_points.row(c_bricks(j,1)),
                                        r_points.row(c_bricks(j,2)),
                                        r_points.row(c_bricks(j,3))) \
                                        - pnt.row(i).transpose();


        // the line ends here (relative to the observation point)
        r2 = eval_polygon(0.5, 0.5, r_points.row(c_bricks(j,4)),
                                        r_points.row(c_bricks(j,5)),
                                        r_points.row(c_bricks(j,6)),
                                        r_points.row(c_bricks(j,7))) \
                                        - pnt.row(i).transpose();


        // get the midpoint
        d = -0.5*(r2 + r1);

        // distance
        dist = d.norm();

        // check for near field
        if (dist < near_field_distance){
          
          //  my_cnt_nf += 1;

          // n direction
          for (int l = 0; l < N_n[j]; ++l){


            // the strand u coordinate
            u = (2*l+1.)/(2*N_n[j]);


            // b direction
            for (int m = 0; m < N_b[j]; ++m){

              
              // the strand v coordinate
              v = (2*m+1.)/(2*N_b[j]);

              // the line starts here
              r1 = eval_polygon(u, v, r_points.row(c_bricks(j,0)),
                                              r_points.row(c_bricks(j,1)),
                                              r_points.row(c_bricks(j,2)),
                                              r_points.row(c_bricks(j,3))) \
                                              - pnt.row(i).transpose();


              // the line ends here
              r2 = eval_polygon(u, v, r_points.row(c_bricks(j,4)),
                                              r_points.row(c_bricks(j,5)),
                                              r_points.row(c_bricks(j,6)),
                                              r_points.row(c_bricks(j,7))) \
                                              - pnt.row(i).transpose();

              my_B.row(i) += I_str(j)*mfd_integral_current_segment(r1, r2).transpose();


            }


          }

        }

        else{
          
          my_B.row(i) += I_str(j)*N_n[j]*N_b[j]*mfd_integral_current_segment(r1, r2).transpose();
        }

      }
    }
    }


    
    #pragma omp critical
    {

      B += my_B;

    }
  }

  // apply factor mu_0/(4 pi)
  B *= 1e-7;
  
  //  copy data to return pointers
  to_c_array(B, ret_data_B);

  return;

}


double compute_L(const double *segs, const int num_segs, const double radius, const int num_points, const int is_open){
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

  std::cout << "********************************" << std::endl;
  std::cout << "  compute inductance" << std::endl;
  std::cout << "********************************" << std::endl;

  // for time measurement
  std::chrono::time_point<std::chrono::steady_clock> t_start, t_end;
  std::chrono::duration<double> t_el;
  
  //  make an eigen style matrix for the segments
  Eigen::MatrixXd segs_mat = build_MatrixXd(segs, num_segs, 6);

  t_start = std::chrono::steady_clock::now();

  double L = compute_self_inductance(segs_mat, radius, num_points, is_open);
	
  t_end = std::chrono::steady_clock::now();
  t_el = t_end - t_start;

  std::cout << "elapsed time = " << t_el.count() << " sec." << std::endl;

  return L;

}

void compute_A_iron(double *ret_A,
                    const int num_pnt,
                    const int num_src,
                    const double *pnt,
                    const double *q,
                    const double *n,
                    const double *w,
                    const double *A, 
                    const double *dA){
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
  
  // set zero
  for(int i = 0; i < 3*num_pnt; ++i){
    ret_A[i] = 0.0;
  }

  #pragma omp parallel
  {

    double dx, dy, dz;
    double nd;
    double dist;
    double f1, f2;
    double *my_A = (double *) calloc(3*num_pnt, sizeof(double));
    double pi4 = 4.0*PI;

    for(int i = 0; i < num_pnt; ++i){
      #pragma omp single nowait

      {
        for(int j = 0; j < num_src; ++j){


          dx = pnt[3*i] - q[3*j];
          dy = pnt[3*i + 1] - q[3*j + 1];
          dz = pnt[3*i + 2] - q[3*j + 2];
          dist = std::sqrt(dx*dx + dy*dy + dz*dz);
          nd = n[3*j]*dx + n[3*j + 1]*dy + n[3*j + 2]*dz;

          f1 = w[j]/dist/pi4;
          f2 = nd*w[j]/dist/dist/dist/pi4;

          my_A[3*i] -= f1*dA[3*j] + f2*A[3*j];
          my_A[3*i + 1] -= f1*dA[3*j + 1] + f2*A[3*j + 1];
          my_A[3*i + 2] -= f1*dA[3*j + 2] + f2*A[3*j + 2];

        }
      }
    }
    #pragma omp critical
    {
      for(int i = 0; i < 3*num_pnt; ++i){
        ret_A[i] += my_A[i];
      }
    }
  }
}

void compute_B_iron(double *ret_B,
                    const int num_pnt,
                    const int num_src,
                    const double *pnt,
                    const double *q,
                    const double *n,
                    const double *w,
                    const double *A, 
                    const double *dA){
/**
    * Compute the magnetic flux density, due to a magnetized
    * iron yoke. This function takes all integration points, normal vectors
    * and weights, as well as the vector potentials and normal derivatives
    * evaluated at the integration points.
    *  
    * @param ret_B The magnetic flux density.
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
  
  // set zero
  for(int i = 0; i < 3*num_pnt; ++i){
    ret_B[i] = 0.0;
  }

  //std::cout << "A = " << A[0] << " , " << A[1] << " , " << A[2] << std::endl;
  #pragma omp parallel
  {

    double dx, dy, dz;
    double nd;
    double dist;
    double dist_3, dist_5;
    double f1[3], f2[3];
    double *my_B = (double *) calloc(3*num_pnt, sizeof(double));
    double pi4 = 4.0*PI;

    for(int i = 0; i < num_pnt; ++i){

      #pragma omp single nowait
      {
        for(int j = 0; j < num_src; ++j){
            
          dx = pnt[3*i] - q[3*j];
          dy = pnt[3*i + 1] - q[3*j + 1];
          dz = pnt[3*i + 2] - q[3*j + 2];
          dist = std::sqrt(dx*dx + dy*dy + dz*dz);
          dist_3 = dist*dist*dist;
          dist_5 = dist_3*dist*dist;
          nd = n[3*j]*dx + n[3*j + 1]*dy + n[3*j + 2]*dz;

          f1[0] = w[j]*dx/dist_3/pi4;
          f1[1] = w[j]*dy/dist_3/pi4;
          f1[2] = w[j]*dz/dist_3/pi4;

          f2[0] = -w[j]*(n[3*j]/dist_3 - 3.0*nd*dx/dist_5)/pi4;
          f2[1] = -w[j]*(n[3*j + 1]/dist_3 - 3.0*nd*dy/dist_5)/pi4;
          f2[2] = -w[j]*(n[3*j + 2]/dist_3 - 3.0*nd*dz/dist_5)/pi4;

          my_B[3*i] += f1[1]*dA[3*j + 2] - f2[1]*A[3*j + 2] - f1[2]*dA[3*j + 1] + f2[2]*A[3*j + 1];
          my_B[3*i + 1] += f1[2]*dA[3*j + 0] - f2[2]*A[3*j + 0] - f1[0]*dA[3*j + 2] + f2[0]*A[3*j + 2];
          my_B[3*i + 2] += f1[0]*dA[3*j + 1] - f2[0]*A[3*j + 1] - f1[1]*dA[3*j + 0] + f2[1]*A[3*j + 0];


        }
      }
    }
    #pragma omp critical
    {
      for(int i = 0; i < 3*num_pnt; ++i){
        ret_B[i] += my_B[i];
      }
    }
  }
}

/*
void compute_dl_field_design_matrix(double *ret_mat,
                                    const int num_pnt,
                                    const int num_nodes,
                                    const int num_el,
                                    const int num_quad,
                                    const double *r,
                                    const double *p,
                                    const int *c,
                                    const int *element_types){

  // check if we need to zero the matrix

  #pragma omp parallel
  {

    double dx, dy, dz;
    double nd;
    double dist;
    double dist_3, dist_5;
    double f1[3], f2[3];
    double *my_mat = (double *) calloc(9*num_nodes*num_pnt, sizeof(double));
    double pi4 = 4.0*PI;
    double q[3*num_quad];

    // loop over all evaluation points
    for(int i = 0; i < num_pnt; ++i){

      #pragma omp single nowait
      {

        // loop over all elements
        for(int j = 0; j < num_el; ++j){

          // evaluate the boundary element

          // get the points q, the normals n and the weights w

          // evaluate the kernel function

          // evaluate the basis functions

          // zero the integration values

          // loop over all integration points
          for(int k = 0; k < num_quad; ++k){

            // loop over all basis functions
            for(int l = 0; l < element_types[j], ++l){

              // increment the integration values

            }

          }
          // sort into the design matrix



        }
      }
    }
    #pragma omp critical
    {
      for(int i = 0; i < 9*num_nodes*num_pnt; ++i){
        ret_mat[i] += my_mat[i];
      }
    }
  }

}

void evaluate_T6(){



}
*/

void compute_B_eddy_ring(double *ret_B,
                    const int num_pnt,
                    const int num_src,
                    const double *pnt,
                    const double *q,
                    const double *s,
                    const double *w){
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

  // set zero
  for(int i = 0; i < 3*num_pnt; ++i){
    ret_B[i] = 0.0;
  }

  #pragma omp parallel
  {

    double dx, dy, dz;
    double dist;
    double dist_3;
    double *my_B = (double *) calloc(3*num_pnt, sizeof(double));
    double pi4 = 4.0*PI;

    for(int i = 0; i < num_pnt; ++i){

      #pragma omp single nowait
      {
        for(int j = 0; j < num_src; ++j){
            
          dx = pnt[3*i] - q[3*j];
          dy = pnt[3*i + 1] - q[3*j + 1];
          dz = pnt[3*i + 2] - q[3*j + 2];
          dist = std::sqrt(dx*dx + dy*dy + dz*dz);
          dist_3 = dist*dist*dist;

          // 4 pi missing here. It is applied below.
          my_B[3*i]     += w[j]*(dz*s[3*j+1] - dy*s[3*j+2])/dist_3;
          my_B[3*i + 1] += w[j]*(dx*s[3*j+2] - dz*s[3*j  ])/dist_3;
          my_B[3*i + 2] += w[j]*(dy*s[3*j  ] - dx*s[3*j+1])/dist_3;


        }
      }
    }
    #pragma omp critical
    {
      for(int i = 0; i < 3*num_pnt; ++i){
        ret_B[i] += my_B[i]/pi4;
      }
    }
  }             
}

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
                    const double *w){
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

  // set zero (ML: Its anoying that we need to do this. There should be a calloc method in cython too!)
  for(int i = 0; i < num_nodes*num_pnt; ++i){
    ret_mat[i] = 0.0;
  }

  #pragma omp parallel
  {

  // variables for the computation of the inverse distance
  double dx, dy, dz, dist, dist_3;
  // these variables are for more readable code
  double Bx, By, Bz;
  // the factor 4 pi
  double pi4 = 4.0*PI;
  // allocate memory for the working matrix
  double *my_mat = (double *) calloc(num_nodes, sizeof(double));
  // allocate memory for the integration points
  double *r_p = (double *) calloc(3*num_quad, sizeof(double));
  // allocate memory for the basis function and derivative evaluations
  double *curls = (double *) calloc(3*num_quad*num_cell_nodes, sizeof(double));
  // index of the first element in the cell array
  int c_idx;

  // loop over all points
  for (int j = 0; j < num_pnt; ++j){

    #pragma omp single nowait
    {

    // set internal memory zero
    for (int i = 0; i < num_nodes; ++i) my_mat[i] = 0.0;

    // set the cell index
    c_idx = num_cell_nodes*i;

    // compute the integration points
    evaluate_boundary_element(r_p, cells, c_idx, num_cell_nodes, nodes, basis, num_quad);
   
    // compute the curls at the integration points
    evaluate_surface_curls(curls, cells, c_idx, num_cell_nodes, nodes, basis_der, num_quad);

    // loop over all boundary elements 
    for (int i = 0; i < num_cells; ++i){
  
      // loop over all quadrature points
      for (int k = 0; k < num_quad; ++k){

        // the distance vector
        dx = pnt[3*j    ] - r_p[3*k    ];
        dy = pnt[3*j + 1] - r_p[3*k + 1];
        dz = pnt[3*j + 2] - r_p[3*k + 2];

        // the distance
        dist = std::sqrt(dx*dx + dy*dy + dz*dz);

        // the distance cubed
        dist_3 = dist*dist*dist;
        
        // loop over all basis functions
        for (int l = 0; l < num_cell_nodes; ++l){

          // this is to compute the three flux density components
          Bx = dz*curls[3*k*num_cell_nodes + 3*l + 1] - dy*curls[3*k*num_cell_nodes +3*l + 2];
          By = dx*curls[3*k*num_cell_nodes + 3*l + 2] - dz*curls[3*k*num_cell_nodes +3*l    ];
          Bz = dy*curls[3*k*num_cell_nodes + 3*l    ] - dx*curls[3*k*num_cell_nodes +3*l + 1];
          
          // this assembles the observed linear combination
          my_mat[cells[c_idx + l]] += w[k]*(n[3*j]*Bx + n[3*j + 1]*By + n[3*j + 2]*Bz)/pi4/dist_3;

        } // basis functions
      } // quad points
    } // elements

    } // open mp single

    #pragma omp critical
    {
      for(int i = 0; i < num_nodes; ++i){
        ret_mat[j*num_nodes + i] += my_mat[i];
      }
    } // open mp critical

  } // targets

  } // open mp parallel

}


void compute_A_mlfmm(double *A,
                    const int num_tar,
                    const int num_src,
                    const double *src_pts,
                    const double *src_vec,
                    const double *tar_pts,
                    const int L,
                    const int max_tree_lvl,
                    const double *b_box_c,
                    const int potential_spec){
/**
    * Compute the magnetic vector potential with the multilevel fast
    * multipole method.
    *  
    * @param A A pointer to the return data,
    * @param num_tar The number of target points.
    * @param num_src The number of source points.
    * @param src_pts The source points pointer.
    * @param src_vec The source vectors pointer.
                     The source vector includes the normal vectors at the integration points.
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

  // for time measurement
  std::chrono::time_point<std::chrono::steady_clock> t_start, t_end;
  std::chrono::duration<double> t_el;

  //  make eigen matrix objects
  Eigen::MatrixXd p = build_MatrixXd(src_pts, num_src, 3);
  Eigen::MatrixXd g = build_MatrixXd(src_vec, num_src, 6);
  Eigen::MatrixXd tar = build_MatrixXd(tar_pts, num_tar, 3);
  Eigen::MatrixXd b_box = build_MatrixXd(b_box_c, 2, 3);

  // std::cout << " p = " << p << std::endl;
  //std::cout << " g = " << g << std::endl;
  //std::cout << " tar = " << tar << std::endl;
  //std::cout << " b_box = " << b_box << std::endl;

  //make space for the magnetic vector potential
  Eigen::MatrixXd A_mat(num_tar, 3);
  A_mat.setZero();

  // Switch between the potential types
  if (potential_spec == 0){

    // Make a mlfmm object
    Mlfmm<MagneticVectorPotentialMonitor> mlfmm(p, g, tar);

    // meliebsc: ToDo put the following into a template function...

    // set up the domain
    mlfmm.set_bounding_box(b_box);

    // Compute the magnetic vector potential
    std::cout << "computation MLFMM ..." << std::endl;

    t_start =  std::chrono::steady_clock::now();

    A_mat =  mlfmm.compute_A(max_tree_lvl, L, 1);

    t_end =  std::chrono::steady_clock::now();
    t_el = t_end - t_start;

    std::cout << "  elapsed time = " << t_el.count() << " sec." << std::endl;

  }
  else if(potential_spec == 1){

    // Make a mlfmm object
    Mlfmm<VectorSingleLayerMonitor> mlfmm(p, g, tar);

    // meliebsc: ToDo put the following into a template function...
    
    // set up the domain
    mlfmm.set_bounding_box(b_box);

    // Compute the magnetic vector potential
    std::cout << "computation MLFMM ..." << std::endl;

    t_start =  std::chrono::steady_clock::now();

    A_mat =  mlfmm.compute_A(max_tree_lvl, L, 1);

    t_end =  std::chrono::steady_clock::now();
    t_el = t_end - t_start;

    std::cout << "  elapsed time = " << t_el.count() << " sec." << std::endl;
  }
  else if(potential_spec == 2){

    // Make a mlfmm object
    Mlfmm<VectorDoubleLayerMonitor> mlfmm(p, g, tar);

    // meliebsc: ToDo put the following into a template function...
    
    // set up the domain
    mlfmm.set_bounding_box(b_box);

    // Compute the magnetic vector potential
    std::cout << "computation MLFMM ..." << std::endl;

    t_start =  std::chrono::steady_clock::now();

    A_mat =  mlfmm.compute_A(max_tree_lvl, L, 1);

    t_end =  std::chrono::steady_clock::now();
    t_el = t_end - t_start;

    std::cout << "  elapsed time = " << t_el.count() << " sec." << std::endl;
  }


  //  copy data the return pointer
  to_c_array(A_mat, A);

  return;

}

void compute_B_mlfmm(double *B,
                    const int num_tar,
                    const int num_src,
                    const double *src_pts,
                    const double *src_vec,
                    const double *tar_pts,
                    const int L,
                    const int max_tree_lvl,
                    const double *b_box_c,
                    const int potential_spec){
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
    * @param potential_spec An integer specifying the potential kind. Options are: 
                          (0) Curl of Magnetic vector potential (Biot-Savart)
                          (1) Curl of Vector-Valued-Single-Layer (For iron magnetization dA)
                          (2) Curl of Vector-Valued-Double-Layer (For iron magnetization A)
    * @return Nothing.
*/

  // for time measurement
  std::chrono::time_point<std::chrono::steady_clock> t_start, t_end;
  std::chrono::duration<double> t_el;

  //  make eigen matrix objects
  Eigen::MatrixXd p = build_MatrixXd(src_pts, num_src, 3);
  Eigen::MatrixXd g = build_MatrixXd(src_vec, num_src, 6);
  Eigen::MatrixXd tar = build_MatrixXd(tar_pts, num_tar, 3);
  Eigen::MatrixXd b_box = build_MatrixXd(b_box_c, 2, 3);

  // std::cout << " b_box = " << b_box << std::endl;

  //make space for the magnetic vector potential
  Eigen::MatrixXd B_mat(num_tar, 3);
  B_mat.setZero();

  // Switch between the potential types
  if (potential_spec == 0){

    // Make a mlfmm object
    Mlfmm<MagneticFluxDensityMonitor> mlfmm(p, g, tar);

    // meliebsc: ToDo put the following into a template function...

    // set up the domain
    mlfmm.set_bounding_box(b_box);

    // Compute the magnetic vector potential
    std::cout << "computation MLFMM ..." << std::endl;

    t_start =  std::chrono::steady_clock::now();

    B_mat =  mlfmm.compute_A(max_tree_lvl, L, 1);

    t_end =  std::chrono::steady_clock::now();
    t_el = t_end - t_start;

    std::cout << "  elapsed time = " << t_el.count() << " sec." << std::endl;

  }
  else if(potential_spec == 1){

    // Make a mlfmm object
    Mlfmm<CurlVectorSingleLayerMonitor> mlfmm(p, g, tar);

    // meliebsc: ToDo put the following into a template function...
    
    // set up the domain
    mlfmm.set_bounding_box(b_box);

    // Compute the magnetic vector potential
    std::cout << "computation MLFMM ..." << std::endl;

    t_start =  std::chrono::steady_clock::now();

    B_mat =  mlfmm.compute_A(max_tree_lvl, L, 1);

    t_end =  std::chrono::steady_clock::now();
    t_el = t_end - t_start;

    std::cout << "  elapsed time = " << t_el.count() << " sec." << std::endl;
  }
  else if(potential_spec == 2){

    // Make a mlfmm object
    Mlfmm<CurlVectorDoubleLayerMonitor> mlfmm(p, g, tar);

    // meliebsc: ToDo put the following into a template function...
    
    // set up the domain
    mlfmm.set_bounding_box(b_box);

    // Compute the magnetic vector potential
    std::cout << "computation MLFMM ..." << std::endl;

    t_start =  std::chrono::steady_clock::now();

    B_mat =  mlfmm.compute_A(max_tree_lvl, L, 1);

    t_end =  std::chrono::steady_clock::now();
    t_el = t_end - t_start;

    std::cout << "  elapsed time = " << t_el.count() << " sec." << std::endl;
  }


  //  copy data the return pointer
  to_c_array(B_mat, B);

  return;

}


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
                    const double *b_box_c){
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

  // for time measurement
  std::chrono::time_point<std::chrono::steady_clock> t_start, t_end;
  std::chrono::duration<double> t_el;

  //  make eigen matrix objects
  Eigen::MatrixXd p_coil = build_MatrixXd(src_pts_coil, num_src_coil, 3);
  Eigen::MatrixXd g_coil = build_MatrixXd(src_vec_coil, num_src_coil, 6);

  Eigen::MatrixXd p_iron = build_MatrixXd(src_pts_iron, num_src_iron, 3);
  Eigen::MatrixXd g_A = build_MatrixXd(src_vec_A, num_src_iron, 6);
  Eigen::MatrixXd g_dA = build_MatrixXd(src_vec_dA, num_src_iron, 6);

  Eigen::MatrixXd tar = build_MatrixXd(tar_pts, num_tar, 3);
  Eigen::MatrixXd b_box = build_MatrixXd(b_box_c, 2, 3);

  // std::cout << " b_box = " << b_box << std::endl;

  // ====================================================
  // Compute the fields
  // ====================================================
  
  t_start =  std::chrono::steady_clock::now();

  Mlfmm<MagneticFluxDensityMonitor> fmm(p_coil, g_coil, tar);
  fmm.set_bounding_box(b_box);

  Mlfmm<CurlVectorSingleLayerMonitor> fmm_dA(p_iron, g_dA, tar);
  fmm_dA.set_bounding_box(b_box);

  Mlfmm<CurlVectorDoubleLayerMonitor> fmm_A(p_iron, g_A, tar);
  fmm_A.set_bounding_box(b_box);

  // evaluate B fields
  // coil
  Eigen::MatrixXd B_mat = fmm.compute_A(max_tree_lvl, L, 1);

  // Single layer potential
  B_mat += -1e7/4.0/PI*fmm_dA.compute_A(max_tree_lvl, L, 1);

  // Double layer potential
  B_mat += 1e7/4.0/PI*fmm_A.compute_A(max_tree_lvl, L, 1);

  t_end =  std::chrono::steady_clock::now();
  t_el = t_end - t_start;

  std::cout << "  elapsed time = " << t_el.count() << " sec." << std::endl;

  //  copy data the return pointer
  to_c_array(B_mat, B);

  // ====================================================
  // To do:
  //
  // Most of the work in this function is used for the
  // cluster tree generation.
  // This is done 3 times here. Possible improvements:
  //
  // - Write a combined monitor for the all sources that
  //   can be switched in type
  //
  // - Then write a function to add leaf moments with new
  //   sources.
  // ====================================================




  return;

}

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
                     const double near_field_distance){
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

  std::cout << "***************************************" << std::endl;
  std::cout << "  solid harmonic expansion calculator  " << std::endl;
  std::cout << "***************************************" << std::endl;

  // for time measurement
  std::chrono::time_point<std::chrono::steady_clock> t_start, t_end;
  std::chrono::duration<double> t_el;

  // See the technical documentation for the definition of the input files

  //  make eigen objects
  Eigen::MatrixXd c_points = build_MatrixXd(p, num_points, 3);
  Eigen::VectorXd I_str = build_VectorXd(I_strand, num_bricks);
  Eigen::MatrixXi c_bricks = build_MatrixXi(c, num_bricks, 8);
  Eigen::MatrixXd pnt = build_MatrixXd(tar_ptr, num_tar, 3);


  // total number of multipole moment coefficients
  int num_coeffs = (int) (L+1)*(L+1);

  // print out the details
  std::cout << "number of conductor segments = " << num_bricks << std::endl;
  std::cout << "number of expansion points = " << num_tar << std::endl;
  std::cout << "number of expansion coefficients (per point) = " << num_coeffs << std::endl;
  // std::cout << "number of interactions = " << num_tar*num_src << std::endl;  // this most certaintly overflows

  // get the quadrature points and weights
  Eigen::MatrixXd Q = get_1D_gauss_integration_points(num_quad_points);

  // make space for results
  Eigen::MatrixXcd M(3*num_tar, num_coeffs);          // multipole moments, the result
  M.setZero();
  //  in the M matrix we store the three components Mx,My,Mz in the rows, for all evaluation points
  //   row 1:  Mx(p_1)_0^0  ,  Mx(p_1)_1^-1   ,  Mx(p_1)_1^0   ,  Mx(p_1)_1^1  , ...  
  //   row 2:  My(p_1)_0^0  ,  My(p_1)_1^-1   ,  My(p_1)_1^0   ,  My(p_1)_1^1  , ...  
  //   row 3:  Mz(p_1)_0^0  ,  Mz(p_1)_1^-1   ,  Mz(p_1)_1^0   ,  Mz(p_1)_1^1  , ...
  //   row 4:  Mx(p_2)_0^0  ,  Mx(p_2)_1^-1   ,  Mx(p_2)_1^0   ,  Mx(p_2)_1^1  , ...  
  //   row 5:  My(p_2)_0^0  ,  My(p_2)_1^-1   ,  My(p_2)_1^0   ,  My(p_2)_1^1  , ...  
  //   row 6:  Mz(p_2)_0^0  ,  Mz(p_2)_1^-1   ,  Mz(p_2)_1^0   ,  Mz(p_2)_1^1  , ...

  std::cout << "start computation" << std::endl;

  t_start =  std::chrono::steady_clock::now();

  #pragma omp parallel
  {
    // make space for the results of this process
    Eigen::MatrixXcd my_M(3*num_tar,num_coeffs);
    my_M.setZero();

    // make space for auxilliary vectors
    Eigen::Vector3d r1,r2,c,d;

    // this is a container for the difference vectors,
    //  i.e. the brick corners minus the observation points
    Eigen::MatrixXd r_points = c_points;

    // distance between brick center and expansion point
    double dist;

    // local brick coordinates
    double u,v;

    for (int i = 0; i < num_tar ; ++i){

      // subtract the ovservation points
      r_points.col(0) = c_points.col(0);// .array() - pnt(i,0);
      r_points.col(1) = c_points.col(1);// .array() - pnt(i,1);
      r_points.col(2) = c_points.col(2);// .array() - pnt(i,2);

      #pragma omp single nowait
      {
      for(int j = 0; j < num_bricks ; ++j){

        // the line starts here
        r1 = eval_polygon(0.5, 0.5, r_points.row(c_bricks(j,0)),
                                        r_points.row(c_bricks(j,1)),
                                        r_points.row(c_bricks(j,2)),
                                        r_points.row(c_bricks(j,3)));

        // the line ends here
        r2 = eval_polygon(0.5, 0.5, r_points.row(c_bricks(j,4)),
                                        r_points.row(c_bricks(j,5)),
                                        r_points.row(c_bricks(j,6)),
                                        r_points.row(c_bricks(j,7)));

        // get the midpoint
        d = pnt.row(i).transpose() - 0.5*(r2 + r1);

        // distance
        dist = d.norm();

        // check for near field  (At the moment, we always compute all sources)
        // if (dist < near_field_distance){
        if(true){

          // n direction
          for (int l = 0; l < N_n[j]; ++l){
            
            // the strand u coordinate
            u = (2*l+1.)/(2*N_n[j]);

            // b direction
            for (int m = 0; m < N_b[j]; ++m){

              // the strand v coordinate
              v = (2*m+1.)/(2*N_b[j]);

              // the line starts here
              r1 = eval_polygon(u, v, r_points.row(c_bricks(j,0)),
                                              r_points.row(c_bricks(j,1)),
                                              r_points.row(c_bricks(j,2)),
                                              r_points.row(c_bricks(j,3)));

              // the line ends here
              r2 = eval_polygon(u, v, r_points.row(c_bricks(j,4)),
                                              r_points.row(c_bricks(j,5)),
                                              r_points.row(c_bricks(j,6)),
                                              r_points.row(c_bricks(j,7)));

              // compute the integral
              my_M.block(3*i,0,3,num_coeffs) += she_line_current(r1,r2,Q,r_ref,L).transpose()*I_str(j);
            }


          }

        }

        else{
              // compute the integral 
              my_M.block(3*i,0,3,num_coeffs) += (she_line_current(r1,r2,Q,r_ref,L).transpose()*I_str(j)*N_n[j]*N_b[j]);
        }

      }
    }
    }
    
    #pragma omp critical
    {

      M += my_M;
    }
  }

  // apply factor mu_0/(8 pi)
  M *= 0.5*1e-7;
  
  t_end =  std::chrono::steady_clock::now();
  t_el = t_end - t_start;

  std::cout << "elapsed time = " << t_el.count() << " sec." << std::endl;

  //  copy data to return pointers
  to_c_array(M.real(), ret_real);
  to_c_array(M.imag(), ret_imag);

  return;

}