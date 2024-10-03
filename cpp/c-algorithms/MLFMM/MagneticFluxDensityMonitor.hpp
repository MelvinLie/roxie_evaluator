#ifndef MLFMM_MFDMONITOR_H_
#define MLFMM_MFDMONITOR_H_

#include <Eigen/Dense>
#include "solid_harmonics.hpp"

 class MagneticFluxDensityMonitor {

 public:
  //constructors
  MagneticFluxDensityMonitor();


  int get_output_space_dimension();

  void computeMoments(Eigen::MatrixXcd *moments, const int L,
                      const Eigen::Vector3d &pos_eval, const Eigen::VectorXd &source_vec);

  Eigen::MatrixXcd evaluateLocalExpansion(const Eigen::Vector3d &pos_eval,
                              const Eigen::Vector3d &cell_center,
                              const int num_multipoles);

  Eigen::Vector3d evaluate_near_field_interaction(const Eigen::Vector3d &r,
                              const Eigen::Vector3d &rp,
                              const Eigen::VectorXd &dr_p);

private:

  int output_space_dimension_ = 3;

 };


#endif
