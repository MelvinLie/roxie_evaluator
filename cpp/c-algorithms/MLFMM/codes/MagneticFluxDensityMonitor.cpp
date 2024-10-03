#ifndef MLFMM_MFDMONITOR_H_
#define MLFMM_MFDMONITOR_H_

#include <Eigen/Dense>



 class MagneticFluxDensityMonitor {

 public:
  //constructors
  MagneticFluxDensityMonitor(){};


  int get_output_space_dimension(){

    return output_space_dimension_;
  }

  Eigen::MatrixXcd evaluateLocalExpansion(const Eigen::Vector3d &pos_eval,
                              const Eigen::Vector3d &cell_center,
                              const int num_multipoles) {



    //number of multipole coefficients
    int num_coeffs = (num_multipoles+1)*(num_multipoles+1);

    //evaluate the derivatives of the solid harmonics
    Eigen::MatrixXcd Rlm_p = Bembel::Rlm_p_alt(num_multipoles, pos_eval, cell_center);

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

  Eigen::Vector3d evaluate_near_field_interaction(const Eigen::Vector3d &r,
                              const Eigen::Vector3d &rp,
                              const Eigen::Vector3d &dr_p) {

      //difference vector
      Eigen::Vector3d diff = r - rp;

    //std::cout << "this interaction = " << dr_p.transpose()/diff.norm() << std::endl;

    return dr_p/diff.norm();
  }

private:

  int output_space_dimension_ = 3;

 };


#endif
