#ifndef SOLID_HARMONICS_H
#define SOLID_HARMONICS_H

/*************************************************************************
Author: Melvin Liebsch
Email: melvin.liebsch@cern.ch
*************************************************************************/

#include <iostream>
#include <fstream>
#include <cmath>  //assoc_legendre

#include <Eigen/Sparse>
#include <Eigen/Core>

/**
* Compute the factorial of an integer number.
*
* @param I The integer as input.
* @return The number I!.
*/
double factorial(int I);

double wigner_d_small(const double theta, const int j, const int m, const int mp);

//correction factor. Needed as we are using the Wigner coefficients based on the normal spherical harmonics
double correction_factor(const int n, const int m);

/**
* Compute the legendre polynomials of a certain order m up to
* some degree L. This function exploits the recursive formula and
* should be used for the fast evaluation.
*
* @param L The maximum degree of the polynomials.
* @param m The order of the polynomials.
* @param x The indipendent variable.
* @return The legendre polynomials of this order.
*/
Eigen::VectorXd recursive_legendre(const int L, const int m, const double x);


/**
* Compute the Legendre polynomials based on the recursive formula.
*
* @param L The maximum degree of the polynomials.
* @param x The indipendent variable.
* @return The all legendre polynomials up to degree L.
*/
Eigen::VectorXd compute_legendre_polynomials(const int L, const double x);

Eigen::MatrixXd compute_legendre_polynomials_and_derivatives(const int L, const double theta);

/**
* Evaluate the irregular solid harmonics S_lm in spherical coordinates
*
* @param L The maximum degree of the expansion.
* @param r The radial coordinate.
* @param theta The polar angle.
* @param phi The azimuth angle.
* @param r_ref A reference radius.
* @return The irregular solid harmonics up to degree L.
*/
Eigen::VectorXcd Slm(const int L, const double r, const double theta, const double phi, const double r_ref = 1);

/**
* Evaluate the rregular solid harmonics S_lm around the Cartesian coordinate y at x.
*
* @param L The maximum degree of the expansion.
* @param x The observation point.
* @param y  The expansion point.
* @param r_ref A reference radius.
* @return The irregular solid harmonics up to degree L.
*/
Eigen::VectorXcd Slm(const int L, const Eigen::Vector3d &x, const Eigen::Vector3d &y, const double r_ref = 1);


/**
* Expand the vector potential due to a line current into
* solid harmonics.
*
* @param r_1 The start position of the line current.
* @param r_2 The end position of the line current.
* @param Q The quadrature points in an eigen Matrix.
* @param r_ref The reference radius.
* @param L The maximum degree of the expansion.
* @param r_0 The observation point.
* @return The solid harmonic expansion at this point.
*/
Eigen::MatrixXcd she_line_current(const Eigen::Vector3d &r_1,
                                  const Eigen::Vector3d &r_2,
                                  const Eigen::MatrixXd &Q,
                                  const double r_ref,
                                  const int L,
                                  const Eigen::Vector3d &r_0 = Eigen::Vector3d::Zero());

Eigen::SparseMatrix<std::complex<double>> make_rotation_matrix(const int num_multipoles,
                                                                const double theta, 
                                                                const double phi);

//Regular solid harmonics R_lm in spherical coordinates
//We follow the definitions in
//Yijun Liu. Fast Multipole Boundary Element Method: Theory and Applications in Engineering. Cambridge University Press, 2009. doi: 10.1017/CBO9780511605345.
//rendering easy formulations for the shift relations
Eigen::VectorXcd Rlm(const int L, const double r, const double theta, const double phi);

//Regular solid harmonics R_lm in spherical coordinates
//We follow the definitions in
//Yijun Liu. Fast Multipole Boundary Element Method: Theory and Applications in Engineering. Cambridge University Press, 2009. doi: 10.1017/CBO9780511605345.
//rendering easy formulations for the shift relations
Eigen::VectorXcd Rlm(const int L, const Eigen::Vector3d &x);

double A_nm(const int n, const int m);

double associated_legendre(const int l, const int m, const double x);

//Here we compute the derivative of the associated legendre polynomial linked
//with the polar angle theta by: P_l^m \circ \cos(\theta)
double associated_legendre_derivative(const int l, const int m, const double theta);

std::complex<double> Ynm_alt(const double theta, const double phi, const int n, const int m);

//Regular solid harmonics R_lm in spherical coordinates
Eigen::VectorXcd Rlm_alt(const int L, const Eigen::Vector3d &x, const double r_ref = 1.);

//Spatial derivatives of the solid harmonics R_lm in spherical coordinates
// this function returns the harmonics in reverse order! Such that the matrix
// can directly be used for multipole expansions
Eigen::MatrixXcd Rlm_p_rpt_alt(const int L, const double r, const double theta, const double phi, const bool reverse_m = false);

//Spatial derivatives of the solid harmonics R_lm
Eigen::MatrixXcd Rlm_p_alt(const int L, const Eigen::Vector3d &x, const Eigen::Vector3d &y, const bool reverse_m = false);

# endif