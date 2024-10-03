#include "MultipoleMomentContainer.hpp"

int get_rot_number(const Eigen::VectorXd &d){

      int ret_val = -1;

      if(d(1) > 0){
        if(d(0) > 0)   ret_val = 0;
        else           ret_val = 1;
      }
      else{
        if(d(0) > 0)   ret_val = 3;
        else           ret_val = 2;
      }
      if(d(2) < 0) ret_val += 4;

      return ret_val;
}


MultipoleMomentContainer::MultipoleMomentContainer(){

    perm_ << 1 ,1 ,1,
             1,  1, -1,
             1, -1, -1,
             1, -1,  1,
            -1,  1,  1,
            -1,  1, -1,
            -1, -1, -1,
            -1, -1,  1;

};

void MultipoleMomentContainer::fill(const int num_multipoles, const int max_level, const Eigen::Matrix<double,2,3> &bbox, int type){

    num_multipoles_ = num_multipoles;

    //fill rotation matrices
    fill_rotation_matrices(num_multipoles);

    //fill transfer matrices
    transfer_matrices_.resize(max_level+1);

    Eigen::Vector3d center = 0.5*(bbox.row(0)+bbox.row(1)).transpose();
    Eigen::Vector3d dir_1 = 0.5*(bbox.row(1).transpose()-center);

    r_0_ = dir_1.norm();

    double this_r = r_0_;

    for(int l = 0; l <= max_level; ++l){

      transfer_matrices_[l] = make_transfer_matrix(num_multipoles_,this_r, type);

      //refine
      this_r *= 0.5;

    }

}

Eigen::SparseMatrix<std::complex<double>> MultipoleMomentContainer::get_moment_matrix(const int level, const int index){

  return moment_matrices_[level][index];

};

Eigen::SparseMatrix<std::complex<double>> MultipoleMomentContainer::get_moment_matrix(const int level, const Eigen::Vector3d d){

    double r = d.norm();

    Eigen::MatrixXd proj = perm_*d/r;

    int index = 0;

    for (int i = 0 ; i < 8 ; ++i){
      if(proj(i,0) >= 1. - 1e-12 ){
        index = i;
        break;
      }
    }

    return get_moment_matrix(level, index);

};

Eigen::SparseMatrix<std::complex<double>> MultipoleMomentContainer::get_moment_matrix(const Eigen::Vector3d d){//const int level,

    double r = d.norm();

    int level = std::lround(std::log2(r_0_/r)+1);

    Eigen::MatrixXd proj = perm_*d/r;

    int index = 0;

    for (int i = 0 ; i < 8 ; ++i){
      if(proj(i,0) >= 1. - 1e-12 ){
        index = i;
        break;
      }
    }

  return get_moment_matrix(level, index);

};

// d vector points FROM father TO son
Eigen::VectorXcd MultipoleMomentContainer::m2m(const int level,const Eigen::Vector3d &d, const Eigen::VectorXcd &moments_in, const bool transpose){

  int rot_indx = get_rot_number(d);

  //rotate
  Eigen::VectorXcd out = rotation_matrices_[rot_indx] * moments_in;
  //transfer
  if(transpose){
    out = (transfer_matrices_[level].adjoint() * out).eval();
  }
  else{
    out = (transfer_matrices_[level] * out).eval();
  }
  //rotate back
  out = (rotation_matrices_[rot_indx].adjoint() * out).eval();


  return out;

}

//level is to be interpreted as the higher level of the interacting cells
void MultipoleMomentContainer::m2m(const int level, const Eigen::Vector3d &d, const Eigen::VectorXcd &moments_in, Eigen::VectorXcd *moments_out, const bool transpose){

    int rot_indx = get_rot_number(d);


    //rotate
    Eigen::VectorXcd out = rotation_matrices_[rot_indx] * moments_in;
    //transfer
    if(transpose){
      out = (transfer_matrices_[level].adjoint() * out).eval();
    }
    else{
      out = (transfer_matrices_[level] * out).eval();
    }
    //rotate back
    out = (rotation_matrices_[rot_indx].adjoint() * out).eval();

    (*moments_out) += out;

    return;

}

//level is to be interpreted as the higher level of the interacting cells
void MultipoleMomentContainer::m2m_mod(const int level,const Eigen::Vector3d &d, const Eigen::MatrixXcd &moments_in, Eigen::MatrixXcd *moments_out, const bool transpose){

    int rot_indx = get_rot_number(d);


    //rotate
    Eigen::MatrixXcd out = rotation_matrices_[rot_indx] * moments_in;
    //transfer
    if(transpose){
      out = (transfer_matrices_[level].adjoint() * out).eval();
    }
    else{
      out = (transfer_matrices_[level] * out).eval();
    }
    //rotate back
    out = (rotation_matrices_[rot_indx].adjoint() * out).eval();

    (*moments_out) += out;

    return;

}

// d vector points FROM son TO father
Eigen::VectorXcd MultipoleMomentContainer::l2l(const int level,const Eigen::Vector3d &d, const Eigen::VectorXcd &locals_in, const bool transpose){

    int rot_indx = get_rot_number(d);

    //rotate
    Eigen::VectorXcd out = rotation_matrices_[rot_indx] * locals_in;
    //transfer
    if(transpose){
      out = (transfer_matrices_[level].adjoint() * out).eval();
    }
    else{
      out = (transfer_matrices_[level] * out).eval();
    }
    //rotate back
    out = (rotation_matrices_[rot_indx].adjoint() * out).eval();


    return out;

}

// d vector points FROM son TO father
void MultipoleMomentContainer::l2l(const int level,const Eigen::Vector3d &d, const Eigen::VectorXcd &locals_in, Eigen::VectorXcd *locals_out, const bool transpose){


    int rot_indx = get_rot_number(d);

    //rotate
    Eigen::VectorXcd out = rotation_matrices_[rot_indx] * locals_in;
    //transfer
    if(transpose){
      out = (transfer_matrices_[level].adjoint() * out).eval();
    }
    else{
      out = (transfer_matrices_[level] * out).eval();
    }
    //rotate back
    out = (rotation_matrices_[rot_indx].adjoint() * out).eval();

    (*locals_out) += out;

    return;

}

// d vector points FROM son TO father
void MultipoleMomentContainer::l2l_mod(const int level,const Eigen::Vector3d &d, const Eigen::MatrixXcd &locals_in, Eigen::MatrixXcd *locals_out, const bool transpose){


    int rot_indx = get_rot_number(d);

    //rotate
    Eigen::MatrixXcd out = rotation_matrices_[rot_indx] * locals_in;
    //transfer
    if(transpose){
      out = (transfer_matrices_[level].adjoint() * out).eval();
    }
    else{
      out = (transfer_matrices_[level] * out).eval();
    }
    //rotate back
    out = (rotation_matrices_[rot_indx].adjoint() * out).eval();

    (*locals_out) += out;


    return;

}

void MultipoleMomentContainer::fill_rotation_matrices(const int num_multipoles){

    rotation_matrices_.resize(8);

    double polar_angle = std::acos(1./std::sqrt(3));


    std::vector<double> thetas = {polar_angle,PI-polar_angle};
    std::vector<double> phis = {0.25*PI,0.75*PI,1.25*PI,1.75*PI};

    //running index
    int k = 0;

    for(double theta: thetas){

      for(double phi: phis){

        rotation_matrices_[k] = make_rotation_matrix(num_multipoles,theta,phi);

        ++k;
      }
    }

}

void MultipoleMomentContainer::print_rotation_matrices(){

    for(int k = 0; k < 8 ; ++k){

      Eigen::MatrixXcd tmp_mat = Eigen::MatrixXcd(rotation_matrices_[k]);
      std::cout << "------------------------------" << std::endl;
      std::cout << tmp_mat << std::endl;
    }

}

double MultipoleMomentContainer::count_RAM(){

    int count = 0;

    //rotation matrices
    for(int i = 0 ; i < rotation_matrices_.size() ; ++i){

      count += rotation_matrices_[i].nonZeros();
    }
    //transfer matrices
    for(int i = 0 ; i < transfer_matrices_.size() ; ++i){

      count += transfer_matrices_[i].nonZeros();
    }

    return count*8e-9;

}

Eigen::SparseMatrix<std::complex<double>> MultipoleMomentContainer::make_transformation_matrix_source(const Eigen::Vector3d &x_p, const Eigen::Vector3d &x){

      Eigen::VectorXcd R = Rlm(num_multipoles_, x-x_p);

      typedef Eigen::Triplet<std::complex<double>> T;
      std::vector<T> tripletList;
      tripletList.reserve(num_multipoles_*num_multipoles_/2);


      //running indices
      int row = 0;
      int k_p;

      for(int l = 0; l <= num_multipoles_; ++l){
        for(int m = -l; m <= l ; ++m){

          int k = 0;
          for(int l_p = 0; l_p <= l; ++l_p){
            for(int m_p = -l_p; m_p <= l_p ; ++m_p){


              k_p = make_k_s(l_p,m_p,l,m);
              if(k_p > -1){

                //std::cout << "( " << row << " , "  << k_p << " , " << R(k) << " )" <<std::endl;
                tripletList.push_back(T(row , k_p , R(k) ));
              }
              ++k;

            }
          }

          ++row;
        }
      }

      int num_coeffs = (num_multipoles_+1)*(num_multipoles_+1);

      Eigen::SparseMatrix<std::complex<double>> mat(num_coeffs,num_coeffs);


      mat.setFromTriplets(tripletList.begin(), tripletList.end());

      //std::cout << Eigen::MatrixXcd(mat) << std::endl;

      return mat;

}

Eigen::SparseMatrix<std::complex<double>> MultipoleMomentContainer::make_transformation_matrix_local(const Eigen::Vector3d &x_p, const Eigen::Vector3d &x){

        Eigen::VectorXcd R = Rlm(num_multipoles_, x_p-x);

        typedef Eigen::Triplet<std::complex<double>> T;
        std::vector<T> tripletList;
        tripletList.reserve(num_multipoles_*num_multipoles_/2);


        //running indices
        int k = 0;
        int k_s;

        for(int l = 0; l <= num_multipoles_; ++l){
          for(int m = -l; m <= l ; ++m){

            int k_p = 0;
            for(int l_p = 0; l_p <= num_multipoles_; ++l_p){
              for(int m_p = -l_p; m_p <= l_p ; ++m_p){

                //std::cout << "l,m = " << l << " , " << m << std::endl;
                //std::cout << "lp,mp = " << l_p << " , " << m_p << std::endl;

                k_s = make_k_s(l,m,l_p,m_p);

                if(k_s > -1){


                  tripletList.push_back(T(k , k_p , R(k_s) ));
                }
                ++k_p;

              }
            }

            ++k;
          }
        }

      int num_coeffs = (num_multipoles_+1)*(num_multipoles_+1);

      Eigen::SparseMatrix<std::complex<double>> mat(num_coeffs,num_coeffs);

      mat.setFromTriplets(tripletList.begin(), tripletList.end());

      return mat;

}

int MultipoleMomentContainer::make_k(const int bottom, const int top,const int L){

      int k = bottom*bottom + bottom + top;

      if(std::abs(top) > bottom) k = -1.;
      if(top > L) k = -1.;
      if(bottom > L) k = -1.;

      return k;

    }

    int make_k_p(const int l,const int m,const int l_p,const int m_p){

      //bottom index
      int b = l-l_p;
      if(b < 0){
        return -1;
      }

      //top index
      int t = m-m_p;

      if(std::abs(t) > b){
        return -1;
      }

      //k = l**2 + l + m
      return b*b + b + t;


}

int MultipoleMomentContainer::make_k_s(const int l,const int m,const int l_p,const int m_p){

      //bottom index
      int b = l_p-l;
      if(b < 0){
        return -1;
      }

      //top index
      int t = m_p-m;

      if(std::abs(t) > b){
        return -1;
      }

      //k = l**2 + l + m
      return b*b + b + t;


}

Eigen::SparseMatrix<std::complex<double>> MultipoleMomentContainer::make_transfer_matrix(const int num_multipoles,const double dist, int type){

      typedef Eigen::Triplet<std::complex<double>> T;
      std::vector<T> tripletList;
      tripletList.reserve(num_multipoles*num_multipoles);

      //imaginary number as helper
      std::complex<double> I(0.0,1.0);

      if(type == SOURCE){

        for(int j = 0; j <= num_multipoles; ++j){
          for(int k = -j; k <= j; ++k){


            for(int n = 0; n <= j; ++n){

              int col = make_k(j - n, k, num_multipoles);
              int row = make_k(j, k, num_multipoles);


              if((col >= 0) && (row >= 0)){


                std::complex<double> M_jk = A_nm(n,0) * A_nm(j-n,k) * std::pow(dist,n) * Ynm_alt(0.,0.,n,0)/ A_nm(j,k);

                tripletList.push_back(T(row, col , M_jk ));
              }


            }
          }
        }

      }
      else{

        for(int j = 0; j <= num_multipoles; ++j){
          for(int k = -j; k <= j; ++k){


            for(int n = j; n <= num_multipoles; ++n){

              int col = make_k(n, k, num_multipoles);
              int row = make_k(j, k, num_multipoles);


              if((col >= 0) && (row >= 0)){


                std::complex<double> L_jk = A_nm(n-j,0) * A_nm(j,k) * std::pow(dist,n-j) * Ynm_alt(0.,0.,n-j,0) / A_nm(n,k) / std::pow(-1.,n+j);

                tripletList.push_back(T(row, col , L_jk ));
              }


            }
          }
        }
      }

      int num_coeffs = (num_multipoles+1)*(num_multipoles+1);

      Eigen::SparseMatrix<std::complex<double>> mat(num_coeffs,num_coeffs);

      mat.setFromTriplets(tripletList.begin(), tripletList.end());

      return mat;
}
