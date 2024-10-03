#include "VectorDoubleLayerMonitor.hpp"

//constructors
VectorDoubleLayerMonitor::VectorDoubleLayerMonitor(){};


int VectorDoubleLayerMonitor::get_output_space_dimension(){

    return output_space_dimension_;
}

void VectorDoubleLayerMonitor::computeMoments(Eigen::MatrixXcd *moments,
                                                    const int L,
                                                    const Eigen::Vector3d &pos_eval,
                                                    const Eigen::VectorXd &source_vec){
  

  //number of multipole coefficients
  int num_coeffs = (L + 1)*(L + 1);

  //evaluate the derivatives of the solid harmonics
  Eigen::MatrixXcd Rlm_p = Rlm_p_alt(L, pos_eval, Eigen::Vector3d::Zero());


  //compute the scalar products n.grad(R)
  Eigen::VectorXcd ndR = source_vec(3) * Rlm_p.col(0) + source_vec(4) * Rlm_p.col(1) + source_vec(5) * Rlm_p.col(2);

  // increment the moments
  moments->col(0) += ndR.conjugate()*source_vec(0);
  moments->col(1) += ndR.conjugate()*source_vec(1);
  moments->col(2) += ndR.conjugate()*source_vec(2);

  return;
}


Eigen::MatrixXcd VectorDoubleLayerMonitor::evaluateLocalExpansion(const Eigen::Vector3d &pos_eval,
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

Eigen::Vector3d VectorDoubleLayerMonitor::evaluate_near_field_interaction(const Eigen::Vector3d &r,
                                                                          const Eigen::Vector3d &rp,
                                                                          const Eigen::VectorXd &dr_p) {

    //difference vector
    Eigen::Vector3d diff = r - rp;

    // the distance
    double dist = diff.norm();

    // the distance cubed
    double dist_3 = dist*dist*dist;

    // the scalar product n.diff
    double nd = dr_p(3)*diff(0) + dr_p(4)*diff(1) + dr_p(5)*diff(2);

    //std::cout << "this interaction = " << dr_p.transpose()/diff.norm() << std::endl;

    return dr_p.segment(0, 3)*nd/dist_3;
}
