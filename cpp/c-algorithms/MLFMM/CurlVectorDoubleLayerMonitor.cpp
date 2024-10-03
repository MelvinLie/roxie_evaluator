#include "CurlVectorDoubleLayerMonitor.hpp"

//constructors
CurlVectorDoubleLayerMonitor::CurlVectorDoubleLayerMonitor(){};


int CurlVectorDoubleLayerMonitor::get_output_space_dimension(){

    return output_space_dimension_;
}

void CurlVectorDoubleLayerMonitor::computeMoments(Eigen::MatrixXcd *moments,
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


Eigen::MatrixXcd CurlVectorDoubleLayerMonitor::evaluateLocalExpansion(const Eigen::Vector3d &pos_eval,
                              const Eigen::Vector3d &cell_center,
                              const int num_multipoles) {


    //number of multipole coefficients
    int num_coeffs = (num_multipoles+1)*(num_multipoles+1);

    //evaluate the derivatives of the solid harmonics
    Eigen::MatrixXcd Rlm_p = Rlm_p_alt(num_multipoles, pos_eval, cell_center);

    //output space dimension
    int dim_output = 3;
    //number of fm components
    int num_fmm_components = 3;

    //fill return vector
    Eigen::MatrixXcd ret_mat(3*num_coeffs, 3);
    ret_mat.setZero();

    /*
              |  0      ,  -Rz.T  ,  Ry.T  |   | Mx |
    	        |                            |   |    |
    curl A =  |  Rz.T   ,     0   , -Rx.T  | . | My |
              |                            |   |    |
              | -Ry.T   ,    Rx.T ,    0   |   | Mz |
    */
    ret_mat.block(0, 1, num_coeffs, 1) = -1.*Rlm_p.col(2);
    ret_mat.block(0, 2, num_coeffs, 1) =     Rlm_p.col(1);

    ret_mat.block(num_coeffs, 0, num_coeffs, 1) =     Rlm_p.col(2);
    ret_mat.block(num_coeffs, 2, num_coeffs, 1) = -1.*Rlm_p.col(0);

    ret_mat.block(2*num_coeffs, 0, num_coeffs, 1) = -1.*Rlm_p.col(1);
    ret_mat.block(2*num_coeffs, 1, num_coeffs, 1) =     Rlm_p.col(0);

    return ret_mat;

}

Eigen::Vector3d CurlVectorDoubleLayerMonitor::evaluate_near_field_interaction(const Eigen::Vector3d &r,
                                                                          const Eigen::Vector3d &rp,
                                                                          const Eigen::VectorXd &dr_p) {

    //difference vector
    Eigen::Vector3d diff = r - rp;

    // the distance
    double dist = diff.norm();

    // the distance cubed
    double dist_3 = dist*dist*dist;

    // the distance to the power of 5
    double dist_5 = dist_3*dist*dist;

    // the scalar product n.diff
    double nd = dr_p(3)*diff(0) + dr_p(4)*diff(1) + dr_p(5)*diff(2);

    // this is the gradient of the kernel
    Eigen::Vector3d grad_k = dr_p.segment(3, 3)/dist_3 - 3.0*nd*diff/dist_5;

    // make a return vector
    Eigen::Vector3d ret_vec;

    // the cross product of grad k x A
    ret_vec(0) = grad_k(1)*dr_p(2) - grad_k(2)*dr_p(1);
    ret_vec(1) = grad_k(2)*dr_p(0) - grad_k(0)*dr_p(2);
    ret_vec(2) = grad_k(0)*dr_p(1) - grad_k(1)*dr_p(0);

    return ret_vec;
}
