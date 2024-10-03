#include "MagneticFluxDensityMonitor.hpp"

//constructors
MagneticFluxDensityMonitor::MagneticFluxDensityMonitor(){};


int MagneticFluxDensityMonitor::get_output_space_dimension(){

    return output_space_dimension_;
}

void MagneticFluxDensityMonitor::computeMoments(Eigen::MatrixXcd *moments,
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


Eigen::MatrixXcd MagneticFluxDensityMonitor::evaluateLocalExpansion(const Eigen::Vector3d &pos_eval,
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
    Eigen::MatrixXcd ret_mat(3*num_coeffs,3);
    ret_mat.setZero();

    /*
              |  0      ,  -Rz.T  ,  Ry.T  |   | Mx |
    	      |                            |   |    |
    curl A =  |  Rz.T   ,     0   , -Rx.T  | . | My |
              |                            |   |    |
              | -Ry.T   ,    Rx.T ,    0   |   | Mz |
    */
    ret_mat.block(0,1,num_coeffs,1) = -1.*Rlm_p.col(2);
    ret_mat.block(0,2,num_coeffs,1) =     Rlm_p.col(1);

    ret_mat.block(num_coeffs,0,num_coeffs,1) =     Rlm_p.col(2);
    ret_mat.block(num_coeffs,2,num_coeffs,1) = -1.*Rlm_p.col(0);

    ret_mat.block(2*num_coeffs,0,num_coeffs,1) = -1.*Rlm_p.col(1);
    ret_mat.block(2*num_coeffs,1,num_coeffs,1) =     Rlm_p.col(0);



    return ret_mat;

}

Eigen::Vector3d MagneticFluxDensityMonitor::evaluate_near_field_interaction(const Eigen::Vector3d &r,
                              const Eigen::Vector3d &rp,
                              const Eigen::VectorXd &dr_p) {

    // make a return vector
    Eigen::Vector3d ret_vec;

    // difference vector
    Eigen::Vector3d diff = r - rp;

    // distance cubed
    double dist_3 = diff.norm()*diff.norm()*diff.norm();
    
    ret_vec(0) = (dr_p(1)*diff(2) - dr_p(2)*diff(1))/dist_3;
    ret_vec(1) = (dr_p(2)*diff(0) - dr_p(0)*diff(2))/dist_3;
    ret_vec(2) = (dr_p(0)*diff(1) - dr_p(1)*diff(0))/dist_3;

    return ret_vec;
}
