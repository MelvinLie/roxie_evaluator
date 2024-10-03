#ifndef MULTIPOLEMOMENTCONTAINER_H_
#define MULTIPOLEMOMENTCONTAINER_H_

#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "solid_harmonics.hpp"

#define PI 3.141592653589793238462643383279

class MultipoleMomentContainer {
 public:

  MultipoleMomentContainer();

  enum Color { SOURCE, LOCAL };

  void fill(const int num_multipoles, const int max_level, const Eigen::Matrix<double,2,3> &bbox, int type = SOURCE);

  Eigen::SparseMatrix<std::complex<double>> get_moment_matrix(const int level,const int index);

  Eigen::SparseMatrix<std::complex<double>> get_moment_matrix(const int level,const Eigen::Vector3d d);

  Eigen::SparseMatrix<std::complex<double>> get_moment_matrix(const Eigen::Vector3d d);

  // d vector points FROM father TO son
  Eigen::VectorXcd m2m(const int level,const Eigen::Vector3d &d, const Eigen::VectorXcd &moments_in, const bool transpose = false);

  //level is to be interpreted as the higher level of the interacting cells
  void m2m(const int level,const Eigen::Vector3d &d, const Eigen::VectorXcd &moments_in, Eigen::VectorXcd *moments_out, const bool transpose = false);

  //level is to be interpreted as the higher level of the interacting cells
  void m2m_mod(const int level,const Eigen::Vector3d &d, const Eigen::MatrixXcd &moments_in, Eigen::MatrixXcd *moments_out, const bool transpose = false);

  // d vector points FROM son TO father
  Eigen::VectorXcd l2l(const int level,const Eigen::Vector3d &d, const Eigen::VectorXcd &locals_in, const bool transpose = false);

  // d vector points FROM son TO father
  void l2l(const int level,const Eigen::Vector3d &d, const Eigen::VectorXcd &locals_in, Eigen::VectorXcd *locals_out, const bool transpose = false);

  // d vector points FROM son TO father
  void l2l_mod(const int level,const Eigen::Vector3d &d, const Eigen::MatrixXcd &locals_in, Eigen::MatrixXcd *locals_out, const bool transpose = false);

  void fill_rotation_matrices(const int num_multipoles);

  void print_rotation_matrices();

  double count_RAM();

  private:

    std::vector<std::vector<Eigen::SparseMatrix<std::complex<double>>,
                Eigen::aligned_allocator<Eigen::SparseMatrix<std::complex<double>> >>> moment_matrices_;

    std::vector<Eigen::SparseMatrix<std::complex<double>>,
                Eigen::aligned_allocator<Eigen::SparseMatrix<std::complex<double>>>> rotation_matrices_;

    std::vector<Eigen::SparseMatrix<std::complex<double>>,
                Eigen::aligned_allocator<Eigen::SparseMatrix<std::complex<double>>>> transfer_matrices_;

    int levels_;
    int num_multipoles_;
    double r_0_;

    Eigen::Matrix<double,8,3> perm_;

    Eigen::SparseMatrix<std::complex<double>> make_transformation_matrix_source(const Eigen::Vector3d &x_p, const Eigen::Vector3d &x);

    Eigen::SparseMatrix<std::complex<double>> make_transformation_matrix_local(const Eigen::Vector3d &x_p, const Eigen::Vector3d &x);

    int make_k(const int bottom, const int top,const int L);

    int make_k_p(const int l,const int m,const int l_p,const int m_p);

    int make_k_s(const int l,const int m,const int l_p,const int m_p);

    Eigen::SparseMatrix<std::complex<double>> make_transfer_matrix(const int num_multipoles,const double dist, int type = SOURCE);

};

#endif
