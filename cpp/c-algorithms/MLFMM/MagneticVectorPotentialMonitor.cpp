#include "MagneticVectorPotentialMonitor.hpp"

//constructors
MagneticVectorPotentialMonitor::MagneticVectorPotentialMonitor(){};


int MagneticVectorPotentialMonitor::get_output_space_dimension(){

  return output_space_dimension_;
}

void MagneticVectorPotentialMonitor::computeMoments(Eigen::MatrixXcd *moments,
                                                    const int L,
                                                    const Eigen::Vector3d &pos_eval,
                                                    const Eigen::VectorXd &source_vec){
  
  //compute the solid Harmonics
  Eigen::VectorXcd R = Rlm_alt(L, pos_eval);

  // increment the moments
  moments->col(0) += R.conjugate()*source_vec(0);
  moments->col(1) += R.conjugate()*source_vec(1);
  moments->col(2) += R.conjugate()*source_vec(2);

  return;
}

Eigen::MatrixXcd MagneticVectorPotentialMonitor::evaluateLocalExpansion(const Eigen::Vector3d &pos_eval,
                              const Eigen::Vector3d &cell_center,
                              const int num_multipoles) {

    //number of multipole coefficients
    int num_coeffs = (num_multipoles+1)*(num_multipoles+1);

    //evaluate solid harmonics
    Eigen::VectorXcd Rlm = Rlm_alt(num_multipoles, pos_eval - cell_center);

    //fill return vector
    Eigen::MatrixXcd ret_mat(output_space_dimension_*num_coeffs,3);
    ret_mat.setZero();

    for(int i = 0; i < output_space_dimension_; ++i){
      ret_mat.block(i*num_coeffs,i,num_coeffs,1)  = Rlm;
    }

    return ret_mat;
}

Eigen::Vector3d MagneticVectorPotentialMonitor::evaluate_near_field_interaction(const Eigen::Vector3d &r,
                              const Eigen::Vector3d &rp,
                              const Eigen::VectorXd &dr_p) {

    //difference vector
    Eigen::Vector3d diff = r - rp;
      

    //std::cout << "this interaction = " << dr_p.transpose()/diff.norm() << std::endl;

    return dr_p.segment(0, 3)/diff.norm();
}
