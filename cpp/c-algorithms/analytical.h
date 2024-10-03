#ifndef ANALYTICAL_H_
#define ANALYTICAL_H_

#include <Eigen/Dense>
#include <Eigen/Core>

/*****************************************
This code provides functions to evaluate analytical
solutions for fields and potentials due to straight line
currents and flat current sheets.

Autor: Melvin Liebsch
email: melvin.liebsch@cern.ch
*****************************************/

/**
  * This function evaluates the magnetic vector potential type integral for a current segment.
  * The points r1 and r2 are the difference vectors from evaluation point to the boundaries of the
  * source segment.
  * The evaluation point here is always considered to be the origin (0,0,0).
  * We evaluate the analytical solution given in "Field computation in accelerator magnets" (5.50)
  * by S. Russenschuck  
  * Notice that the factor mu_o*I/4pi is not applied here!

  * @param r1 The start of the line current segment.
  * @param r2 The end of the line current segment.
  * @param current The current.
  * @return The vector potential at the origin (0, 0, 0).
*/
Eigen::Vector3d mvp_integral_current_segment(const Eigen::Vector3d &r1,
                                          const Eigen::Vector3d &r2,
                                          const double current);

/**
  * This function evaluates the magnetic flux density type integral for a current segment.
  * The points r1 and r2 are the difference vectors from evaluation point to the boundaries of the
  * source segment.
  * The evaluation point here is always considered to be the origin (0,0,0).
  * We evaluate the analytical solution given in "Field computation in accelerator magnets" (5.51)
  * by S. Russenschuck  
  * Notice that the factor mu_o*I/4pi is not applied here!

  * @param r1 The start of the line current segment.
  * @param r2 The end of the line current segment.
  * @return The vector potential at the origin (0, 0, 0).
*/
Eigen::Vector3d mfd_integral_current_segment(const Eigen::Vector3d &r1,
                                          const Eigen::Vector3d &r2);

/**
  * This function evaluates all integrals for a current segment.
  * This means: (magnetic flux density and magnetic vector potential)
  * The points r1 and r2 are the difference vectors from evaluation point to the boundaries of the
  * source segment.
  * The evaluation point here is always considered to be the origin (0,0,0).
  * We evaluate the analytical solution given in "Field computation in accelerator magnets" (5.51)
  * by S. Russenschuck  
  * Notice that the factor mu_o*I/4pi is not applied here!

  * @param r1 The start of the line current segment.
  * @param r2 The end of the line current segment.
  * @param A_mat A pointer to the vector potential matrix.
  * @param B_mat A pointer to the flux density matrix.
  * @param current The current.
  * @param obs_index The index of the observation point in the matrices (the row).
  * @return Nothing.
*/
void compute_integrals_current_segment(const Eigen::Vector3d &r1,
                               const Eigen::Vector3d &r2,
                               Eigen::MatrixXd *A_mat,
                               Eigen::MatrixXd *B_mat,
                               const double current,
                               const int obs_index);


#endif