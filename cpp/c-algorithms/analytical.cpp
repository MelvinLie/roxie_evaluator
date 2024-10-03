#include <Eigen/Dense>
#include <Eigen/Core>
#include <iostream>

Eigen::Vector3d mvp_integral_current_segment(const Eigen::Vector3d &r1,
                                          const Eigen::Vector3d &r2,
                                          const double current){
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

  //the line between the points is defined as r1 + t*(r2 - r1)

  //we denote the vector (r2 - r1) by d!
  Eigen::Vector3d d = r2 - r1;


  //some vector magnitudes
  double d_mag = d.norm();
  double r1_mag = r1.norm();
  double r2_mag = r2.norm();


  //opening angles
  double a_1 = std::asin(r1.dot(d)/r1_mag/d_mag);
  double a_2 = std::asin(r2.dot(d)/r2_mag/d_mag);

  //the factor resulting from integrating 1/cos(alpha)
  double factor = current*std::log(std::sqrt((1-std::sin(a_1))/(1+std::sin(a_1))*(1+std::sin(a_2))/(1-std::sin(a_2))));

  return factor*d/d.norm();

}

Eigen::Vector3d mfd_integral_current_segment(const Eigen::Vector3d &r1,
                                          const Eigen::Vector3d &r2){
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

  //some norms
  double r1_mag = r1.norm();
  double r2_mag = r2.norm();

  //the dot product
  double r1r2 = r1.transpose()*r2;

  //the cross product
  Eigen::Vector3d r1xr2 = r1.cross(r2);

  return (r1_mag + r2_mag)/(r1_mag*r2_mag + r1r2)/r1_mag/r2_mag*r1xr2;

}


void compute_integrals_current_segment(const Eigen::Vector3d &r1,
                               const Eigen::Vector3d &r2,
                               Eigen::MatrixXd *A_mat,
                               Eigen::MatrixXd *B_mat,
                               const double current,
                               const int obs_index){
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

  //the line between the points is defined as r1 + t*(r2 - r1)

  //we denote the vector (r2 - r1) by d!
  Eigen::Vector3d d = r2 - r1;


  //some vector magnitudes
  double d_mag = d.norm();
  double r1_mag = r1.norm();
  double r2_mag = r2.norm();

  //opening angles
  double a_1 = std::asin(r1.dot(d)/r1_mag/d_mag);
  double a_2 = std::asin(r2.dot(d)/r2_mag/d_mag);

  //the factor resulting from integrating 1/cos(alpha)
  double factor = current*std::log(std::sqrt((1-std::sin(a_1))/(1+std::sin(a_1))*(1+std::sin(a_2))/(1-std::sin(a_2))));
  
  //increment the magnetic vector potential matrix
  (*A_mat)(obs_index,0) += factor*d(0)/d.norm();
  (*A_mat)(obs_index,1) += factor*d(1)/d.norm();
  (*A_mat)(obs_index,2) += factor*d(2)/d.norm();

  //the dot product
  double r1r2 = r1.transpose()*r2;

  //the cross product
  Eigen::Vector3d r1xr2 = r1.cross(r2);

  //the factor resulting from integrating cos(alpha)/R**2
  factor = current*(r1_mag + r2_mag)/(r1_mag*r2_mag + r1r2)/r1_mag/r2_mag;

  //increment the magnetic fluix density matrix
  (*B_mat)(obs_index,0) += factor*r1xr2(0);
  (*B_mat)(obs_index,1) += factor*r1xr2(1);
  (*B_mat)(obs_index,2) += factor*r1xr2(2);


  return;

}
