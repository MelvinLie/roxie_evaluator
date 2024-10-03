#include <iostream>
#include <fstream>
#include <cmath>  //assoc_legendre

#include <Eigen/Sparse>
#include <Eigen/Core>

/*************************************************************************
Author: Melvin Liebsch
Email: melvin.liebsch@cern.ch

This code has quite some history.  I started writing it in 2020 for the
evaluation of the magnetic scalar potential using a double layer potential.
The I adapted it to evaluate vector potential and B field due to a distribution
of line currents.

Some of the functions got renamed:
recurse_all_legendre_polynomials --> compute_legendre_polynomials
recursive_legendre_evaluation --> recursive_legendre 

*************************************************************************/




double factorial(int I) {
    /**
    * Compute the factorial of an integer number.
    *
    * @param I The integer as input.
    * @return The number I!.
    */
    double ret_val = 1.;

    for (int i = 2; i < I + 1; ++i) {
        ret_val *= i;
    }
    return ret_val;

}

double wigner_d_small(const double theta, const int j, const int m, const int mp) {

    double fac1 = std::sqrt(factorial(j + m)*factorial(j - m)*factorial(j + mp)*factorial(j - mp));

    int s_min = 0;
    if ((m - mp) > s_min) s_min = m - mp;

    int s_max = j + m;
    if ((j - mp) < s_max) s_max = j - mp;

    double tmp_sum = 0.;

    for (int s = s_min; s <= s_max; ++s) {

        double num = std::pow(-1., mp - m + s) * std::pow(std::cos(0.5*theta), 2 * j + m - mp - 2 * s) * std::pow(std::sin(0.5*theta), mp - m + 2 * s);
        double den = factorial(j + m - s)*factorial(s)*factorial(mp - m + s)*factorial(j - mp - s);

        tmp_sum += num / den;
    }

    return fac1 * tmp_sum;
}

//correction factor. Needed as we are using the Wigner coefficients based on the normal spherical harmonics
double correction_factor(const int n, const int m) {

    double ret_val = std::sqrt(4.*EIGEN_PI / (2.*n + 1));

    if (m < 0) ret_val *= std::pow(-1., m);

    return ret_val;
}

Eigen::VectorXd recursive_legendre(const int L, const int m, const double x) {
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
    //number of coefficients
    int num_coeffs = L + 1 - m;

    //return values
    Eigen::VectorXd ret_vec(num_coeffs);


    //ret_vec(0) = boost::math::legendre_p(m, m, x);            // this function includes the Condon-Shortley phase term
    ret_vec(0) = std::pow(-1., m)*std::assoc_legendre(m, m, x); // the std function does not include the Condon-Shortley phase term

    if (num_coeffs > 1) {
        //ret_vec(1) = boost::math::legendre_p(m + 1, m, x);            // this function includes the Condon-Shortley phase term
        ret_vec(1) = std::pow(-1., m)*std::assoc_legendre(m + 1, m, x); // the std function does not include the Condon-Shortley phase term

        for (int i = 1; i < num_coeffs-1; ++i) {

            //the resutsive relation is:
            // P_{l+1}^m(x) = ((2*l+1)*x*P_{l}^m(x) - (l+m)*P_{l-1}^m(x))/(l-m+1)
            ret_vec(i+1) = ((2*(i + m)+1)*x*ret_vec(i) - (i + 2*m)*ret_vec(i-1))/(i + 1);
            //ret_vec(l+1) = std::pow(-1., m)*std::assoc_legendre(l+1, m, x);
        }
    }


    return ret_vec;

}

Eigen::VectorXd compute_legendre_polynomials(const int L, const double x) {
    /**
    * Compute the Legendre polynomials based on the recursive formula.
    *
    * @param L The maximum degree of the polynomials.
    * @param x The indipendent variable.
    * @return The all legendre polynomials up to degree L.
    */
    //number of coefficients
    int num_coeffs = (L + 1)*(L + 1);

    //return values
    Eigen::VectorXd ret_vec(num_coeffs);

    //temporal container for polynomials
    Eigen::VectorXd p_container;

    for (int m = 0; m <= L; ++m) {

        p_container = recursive_legendre(L, m, x);

        //running variable
        int k = 0;

        for (int l = m; l <= L; ++l) {

            ret_vec(l*l + l + m) = p_container(k);
            if (m > 0) {
                ret_vec(l*l + l - m) = std::pow(-1., m)*factorial(l - m) / factorial(l + m)*ret_vec(l*l + l + m);
            }
            ++k;
        }
    }
    return ret_vec;
}

Eigen::MatrixXd compute_legendre_polynomials_and_derivatives(const int L, const double theta) {

    //argument for legendre polynomial
    double x = std::cos(theta);

    //number of coefficients. We calculate one more to get the derivative
    int num_coeffs = (L + 1)*(L + 1);

    //return values
    Eigen::MatrixXd ret_mat(num_coeffs, 2);
    ret_mat.setZero();

    //temporal container for polynomials
    Eigen::VectorXd p_container;

    //temporal container for derivatives
    Eigen::VectorXd dp_container;

    //Helper vector
    Eigen::VectorXd l_helper;

    //Helper unit array
    Eigen::ArrayXd unit_helper;

    //value of derivative for singularities
    Eigen::VectorXd sing_value;



    for (int m = 0; m <= L; ++m) {

        //legendre polynomials
        p_container = recursive_legendre(L + 1, m, x);

        //array with l counting up
        l_helper = Eigen::VectorXd::LinSpaced(p_container.size(), m, L + 1);


        //Deal with the singularity at \theta = 0
        if (fabs(theta) < 1e-8) {
            if (fabs(m) == 1) {
                sing_value = -0.5*l_helper.array()*(l_helper.array() + 1.);
            }
            else {
                sing_value = 0.*l_helper;
            }
            dp_container = sing_value.segment(0, l_helper.size() - 1);
        }
        //Deal with the singularity at \theta = \pm \pi
        else if (fabs(theta) > EIGEN_PI - 1e-8) {
            if (fabs(m) == 1) {
                unit_helper = -1 * Eigen::ArrayXd::Ones(l_helper.size());

                sing_value = 0.5*unit_helper.pow(l_helper.array() + 1.)*l_helper.array()*(l_helper.array() + 1.);

            }
            else {
                sing_value = 0.*l_helper;
            }
            dp_container = sing_value.segment(0, l_helper.size() - 1);
        }
        else {

            dp_container = (l_helper.segment(0, p_container.size() - 1).array() + 1. - m)*p_container.segment(1, p_container.size() - 1).array()
                - (l_helper.segment(0, p_container.size() - 1).array() + 1.) * x * p_container.segment(0, p_container.size() - 1).array();
            dp_container *= -1.*std::sin(theta) / (x*x - 1.);

        }

        //running variable
        int k = 0;


        for (int l = m; l <= L; ++l) {


            ret_mat(l*l + l + m, 0) = p_container(k);
            ret_mat(l*l + l + m, 1) = dp_container(k);

            if (m > 0) {
                ret_mat(l*l + l - m, 0) = std::pow(-1., m)*factorial(l - m) / factorial(l + m)*ret_mat(l*l + l + m, 0);
                ret_mat(l*l + l - m, 1) = std::pow(-1., m)*factorial(l - m) / factorial(l + m)*ret_mat(l*l + l + m, 1);
            }

            ++k;

        }

    }


    return ret_mat;

}

Eigen::VectorXcd Slm(const int L, const double r, const double theta, const double phi, const double r_ref = 1) {
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

    //number of coefficients
    int num_coeffs = (L + 1)*(L + 1);

    //imaginary number as helper
    std::complex<double> I(0.0, 1.0);

    Eigen::VectorXd p_lm = compute_legendre_polynomials(L, std::cos(theta));

    Eigen::VectorXcd S_lm(num_coeffs);
    S_lm.setZero();

    int k = 0;
    double fac;

    for (int l = 0; l <= L; ++l) {
        for (int m = -l; m <= l; ++m) {

            if (m < 0) {
                fac = std::pow(-1., (double) m)*std::sqrt(factorial(l - m) / factorial(l + m));
            }
            else {
                fac = std::sqrt(factorial(l - m) / factorial(l + m));
            }
            S_lm(k) = std::pow(r_ref/r, (double) l) * fac * p_lm(k)*(std::cos(m*phi) + I * std::sin(m*phi)) / r_ref / r;
            ++k;

        }
    }
    return S_lm;
}

Eigen::VectorXcd Slm(const int L, const Eigen::Vector3d &x, const Eigen::Vector3d &y, const double r_ref = 1) {
    /**
    * Evaluate the rregular solid harmonics S_lm around the Cartesian coordinate y at x.
    *
    * @param L The maximum degree of the expansion.
    * @param x The observation point.
    * @param y  The expansion point.
    * @param r_ref A reference radius.
    * @return The irregular solid harmonics up to degree L.
    */
    Eigen::VectorXd d = x - y;

    double r = d.norm();
    double theta = std::acos(d(2) / r);
    double phi = std::atan2(d(1), d(0));

    return Slm(L, r, theta, phi, r_ref);


}

Eigen::MatrixXcd she_line_current(const Eigen::Vector3d &r_1,
                                    const Eigen::Vector3d &r_2,
                                    const Eigen::MatrixXd &Q,
                                    const double r_ref,
                                    const int L,
                                    const Eigen::Vector3d &r_0 = Eigen::Vector3d::Zero()){
    /**
    * Expand the vector potential due to a line current into
    * solid harmonics.
    *
    * @param r_1 The start position of the line current.
    * @param r_2 The end position of the line current.
    * @param Q The quadrature points in an eigen Matrix.
    * @param r_ref The reference radius.
    * @param L The number of coefficients for the expansion.
    * @param r_0 The observation point.
    * @return The solid harmonic expansion at this point.
    */

    //number of coefficients
    int num_coeffs = (L+1)*(L+1);

    //the integral 1/R
    Eigen::VectorXcd integral(num_coeffs);
    integral.setZero();

    //number of integration points
    int num_q = (int) Q.rows();

    //interation point
    Eigen::Vector3d r;


    //in the next part we perform the Gaussian integration over the line current segment

    //loop over all integration points
    for (int j = 0 ; j < num_q; ++j){

        r = 0.5*(r_2*(Q(j,0) + 1) - r_1*(Q(j,0) - 1));

        integral += Q(j,1)*Slm(L, r, r_0, r_ref);

    }
    //difference vector
    Eigen::Vector3d diff = r_2 - r_1;



    //the return is a matrix of size (num_coeffs x 3).
    //the three columns store the contributions of the (x,y,z) to the multipole moment 
    Eigen::MatrixXcd ret_val(num_coeffs,3);

    

    ret_val.col(0) = diff(0)*integral;
    ret_val.col(1) = diff(1)*integral;
    ret_val.col(2) = diff(2)*integral;


    return ret_val;


}

Eigen::SparseMatrix<std::complex<double>> make_rotation_matrix(const int num_multipoles,
                                                                const double theta, 
                                                                const double phi) {

    typedef Eigen::Triplet<std::complex<double>> T;
    std::vector<T> tripletList;
    tripletList.reserve(num_multipoles*num_multipoles);

    //imaginary number as helper
    std::complex<double> I(0.0, 1.0);

    int l_cum = 0;

    for (int l = 0; l <= num_multipoles; ++l) {
        for (int k = -l; k <= l; ++k) {
            for (int m = -l; m <= l; ++m) {

                double d = wigner_d_small(theta, l, m, k);
                std::complex<double> e = std::cos(k*phi) + I * std::sin(k*phi);

                //Needed as we are using the Wigner coefficients based on the normal spherical harmonics
                double f = correction_factor(l, m) / correction_factor(l, k);


                tripletList.push_back(T(l_cum + m + l, l_cum + k + l, f*d*e));

            }
        }

        l_cum += 2 * l + 1;
    }

    int num_coeffs = (num_multipoles + 1)*(num_multipoles + 1);

    Eigen::SparseMatrix<std::complex<double>> mat(num_coeffs, num_coeffs);

    mat.setFromTriplets(tripletList.begin(), tripletList.end());

    return mat;
}

//Regular solid harmonics R_lm in spherical coordinates
//We follow the definitions in
//Yijun Liu. Fast Multipole Boundary Element Method: Theory and Applications in Engineering. Cambridge University Press, 2009. doi: 10.1017/CBO9780511605345.
//rendering easy formulations for the shift relations
Eigen::VectorXcd Rlm(const int L, const double r, const double theta, const double phi) {

    //number of coefficients
    int num_coeffs = (L + 1)*(L + 1);

    //imaginary number as helper
    std::complex<double> I(0.0, 1.0);


    Eigen::VectorXd p_lm = compute_legendre_polynomials(L, std::cos(theta));

    Eigen::VectorXcd R_lm(num_coeffs);
    R_lm.setZero();

    int k = 0;
    double fac;

    for (int l = 0; l <= L; ++l) {
        for (int m = -l; m <= l; ++m) {

            fac = 1. / factorial(l + m);

            R_lm(k) = fac * p_lm(k)*std::pow(r, l)*(std::cos(m*phi) + I * std::sin(m*phi));

            ++k;

        }
    }

    return R_lm;
}

//Regular solid harmonics R_lm in spherical coordinates
//We follow the definitions in
//Yijun Liu. Fast Multipole Boundary Element Method: Theory and Applications in Engineering. Cambridge University Press, 2009. doi: 10.1017/CBO9780511605345.
//rendering easy formulations for the shift relations
Eigen::VectorXcd Rlm(const int L, const Eigen::Vector3d &x) {

    double r = x.norm();
    double theta = 0.;
    if (r > 1e-12) {
        theta = std::acos(x(2) / r);
    }
    double phi = std::atan2(x(1), x(0));

    //number of coefficients
    int num_coeffs = (L + 1)*(L + 1);

    //imaginary number as helper
    std::complex<double> I(0.0, 1.0);


    Eigen::VectorXd p_lm = compute_legendre_polynomials(L, std::cos(theta));

    Eigen::VectorXcd R_lm(num_coeffs);
    R_lm.setZero();

    int k = 0;
    double fac;

    for (int l = 0; l <= L; ++l) {
        for (int m = -l; m <= l; ++m) {

            fac = 1. / factorial(l + m);

            R_lm(k) = fac * p_lm(k)*std::pow(r, l)*(std::cos(m*phi) + I * std::sin(m*phi));

            ++k;

        }
    }

    return R_lm;
}

double A_nm(const int n, const int m) {

    return std::pow(-1., n) / std::sqrt(factorial(n - m)*factorial(n + m));
}

double associated_legendre(const int l, const int m, const double x) {

    //https://www.osti.gov/servlets/purl/4370075
    double p_lm = 0.;
    int m_abs = m; if (m_abs < 0) m_abs *= -1;
    double x_abs = x; if (x_abs < 0) x_abs *= -1;

    double arg = 1.;//(1.-x_abs)/2.;
    double arg_inc = (1. - x_abs) / 2.;

    double fac_lpn = factorial(l);
    double fac_lmn = fac_lpn;
    double fac_mpn = factorial(m_abs);
    double fac_n = 1.;
    double pow_m1 = 1.;

    p_lm = fac_lpn / fac_lmn * pow_m1 / fac_mpn / fac_n * arg;

    for (int n = 1; n < l + 1; ++n) {

        fac_lpn *= (l + n);
        fac_lmn /= (l - n + 1);
        fac_mpn *= (m_abs + n);
        fac_n *= n;
        arg *= arg_inc;
        pow_m1 *= -1;

        p_lm += fac_lpn / fac_lmn * pow_m1 / fac_mpn / fac_n * arg;


    }
    p_lm *= std::pow((1. - x_abs) / (1. + x_abs), 0.5*m_abs);

    if (x < 0) {
        p_lm *= std::pow(-1., l + m_abs);
    }

    if (m > 0) {


        p_lm *= std::pow(-1.,(double) m_abs)*factorial(l + m) / factorial(l - m);

    }


    return p_lm;

}
//Here we compute the derivative of the associated legendre polynomial linked
//with the polar angle theta by: P_l^m \circ \cos(\theta)
double associated_legendre_derivative(const int l, const int m, const double theta) {
    //https://math.stackexchange.com/questions/3369949/derivative-of-normalized-associated-legendre-function-at-the-limits-of-x-1
    //https://math.stackexchange.com/questions/391672/derivative-of-associated-legendre-polynomials-at-x-pm-1

    double ret_val = 0.;
    double x = std::cos(theta);
    int m_abs = std::abs(m);


    //Deal with singularity at \theta = 0
    if (fabs(theta) < 1e-8) {
        if (m_abs == 1) {
            ret_val = -0.5*l*(l + 1);

            if (m < 0) {


                ret_val *= std::pow(-1., m_abs)*factorial(l - m_abs) / factorial(l + m_abs);

            }
        }
    }
    //Deal with singularity at \theta = \pm \pi
    else if (fabs(theta) > EIGEN_PI - 1e-8) {


        if (m_abs == 1) {
            ret_val = 0.5*std::pow(-1, l + 1)*l*(l + 1);


            if (m < 0) {


                ret_val *= std::pow(-1., m_abs)*factorial(l - m_abs) / factorial(l + m_abs);

            }
        }
    }
    else {

        ret_val = ((l + 1 - m)*associated_legendre(l + 1, m, x) - (l + 1)*x*associated_legendre(l, m, x)) / (x*x - 1);
        ret_val *= -1.*std::sin(theta);
    }


    return ret_val;

}

std::complex<double> Ynm_alt(const double theta, const double phi, const int n, const int m) {

    //imaginary number as helper
    std::complex<double> I(0.0, 1.0);

    double pnm = associated_legendre(n, std::abs(m), std::cos(theta));

    return std::sqrt(factorial(n - std::abs(m)) / factorial(n + std::abs(m))) * pnm * (std::cos(m*phi) + I * std::sin(m*phi));

}

//Regular solid harmonics R_lm in spherical coordinates
Eigen::VectorXcd Rlm_alt(const int L, const Eigen::Vector3d &x, const double r_ref) {


    double r = x.norm();
    double theta = 0.;


    if (r > 1e-12) {
        theta = std::acos(x(2) / r);
    }
    double phi = std::atan2(x(1), x(0));

    //number of coefficients
    int num_coeffs = (L + 1)*(L + 1);

    //imaginary number as helper
    std::complex<double> I(0.0, 1.0);


    Eigen::VectorXd p_lm = compute_legendre_polynomials(L, std::cos(theta));


    Eigen::VectorXcd R_lm(num_coeffs);
    R_lm.setZero();

    int k = 0;
    double fac;

    for (int l = 0; l <= L; ++l) {
        for (int m = -l; m <= l; ++m) {

            if (m < 0) {
                fac = std::pow(-1., m)*std::sqrt(factorial(l - m) / factorial(l + m));
            }
            else {
                fac = std::sqrt(factorial(l - m) / factorial(l + m));
            }

            //R_lm(k) = fac * p_lm(k)*std::pow(r, l)*(std::cos(m*phi) + I * std::sin(m*phi))/std::pow(r_ref, l - 1);
            R_lm(k) = r_ref * fac * p_lm(k)*std::pow(r/r_ref, (double) l)*(std::cos(m*phi) + I * std::sin(m*phi));

            ++k;

        }
    }

    return R_lm;
}

//Spatial derivatives of the solid harmonics R_lm in spherical coordinates
// this function returns the harmonics in reverse order! Such that the matrix
// can directly be used for multipole expansions
Eigen::MatrixXcd Rlm_p_rpt_alt(const int L, const double r, const double theta, const double phi, const bool reverse_m) {

    //number of coefficients
    int num_coeffs = (L + 1)*(L + 1);

    //imaginary number as helper
    std::complex<double> I(0.0, 1.0);


    //Eigen::MatrixXcd sph_harms =  evaluate_harmonic_expansion_fast(L,theta,phi);
    //recusive computation of all polynomials and their first derivatives
    Eigen::MatrixXd p_lm = compute_legendre_polynomials_and_derivatives(L, theta);


    Eigen::MatrixXcd R_rtp(num_coeffs, 3);
    R_rtp.setZero();


    //denominator for R_phi. Again we need to take care of singulatities.
    // From Hospitals rule, we can compute the limit for theta -> 0
    // lim = d_Y_lm_dt / cos(theta) |_theta = 0
    double den_Rp = std::sin(theta);
    int index_Rp = 0;

    if (fabs(theta) < 1e-8) {
        den_Rp = 1.;
        index_Rp = 1;

    }
    else if (fabs(theta) > EIGEN_PI - 1e-8) {
        den_Rp = -1.;
        index_Rp = 1;
    }

    int k = 1;
    

    //double sqrt_4pi = std::sqrt(4.*EIGEN_PI);
    double fac;

    if (reverse_m) {
        for (int l = 1; l <= L; ++l) {

            //fac = sqrt_4pi/std::sqrt(2.*l+1);

            for (int m = l; m >= -l; --m) {

                if (m < 0) {
                    fac = std::pow(-1., m)*std::sqrt(factorial(l - m) / factorial(l + m));
                }
                else {
                    fac = std::sqrt(factorial(l - m) / factorial(l + m));
                }



                R_rtp(k, 0) = fac * l*std::pow(r, l - 1)*p_lm(k, 0)*(std::cos(m*phi) + I * std::sin(m*phi));
                R_rtp(k, 1) = fac * std::pow(r, l - 1)*p_lm(k, 1)*(std::cos(m*phi) + I * std::sin(m*phi));
                R_rtp(k, 2) = fac * m*std::pow(r, l - 1) / den_Rp * p_lm(k, index_Rp)*(-1.*std::sin(m*phi) + I * std::cos(m*phi));


                
                ++k;

            }
        }
    }
    else {
        for (int l = 1; l <= L; ++l) {

            //fac = sqrt_4pi/std::sqrt(2.*l+1);

            for (int m = -l; m <= l; ++m) {

                if (m < 0) {
                    fac = std::pow(-1., m)*std::sqrt(factorial(l - m) / factorial(l + m));
                }
                else {
                    fac = std::sqrt(factorial(l - m) / factorial(l + m));
                }



                R_rtp(k, 0) = fac * l * std::pow(r, l - 1)*p_lm(k, 0)*(std::cos(m*phi) + I * std::sin(m*phi));
                R_rtp(k, 1) = fac * std::pow(r, l - 1)*p_lm(k, 1)*(std::cos(m*phi) + I * std::sin(m*phi));
                R_rtp(k, 2) = fac * m*std::pow(r, l - 1) / den_Rp * p_lm(k, index_Rp)*(-1.*std::sin(m*phi) + I * std::cos(m*phi));

                ++k;

            }
        }
    }



    return R_rtp;


}

//Spatial derivatives of the solid harmonics R_lm
Eigen::MatrixXcd Rlm_p_alt(const int L, const Eigen::Vector3d &x, const Eigen::Vector3d &y, const bool reverse_m = false) {

    //number of coefficients
    int num_coeffs = (L + 1)*(L + 1);


    Eigen::VectorXd d = x - y;

    double r = d.norm();
    double theta = 0.;
    if (r > 1e-10) {
        theta = std::acos(d(2) / r);
    }
    double phi = std::atan2(d(1), d(0));

    //Derivatives in spherical coordinates
    Eigen::MatrixXcd R_rtp = Rlm_p_rpt_alt(L, r, theta, phi, reverse_m);

    Eigen::MatrixXcd ret_mat(num_coeffs, 3);

    ret_mat.col(0) = std::sin(theta)*std::cos(phi)*R_rtp.col(0)
        + std::cos(theta)*std::cos(phi)*R_rtp.col(1)
        - std::sin(phi)*R_rtp.col(2);

    ret_mat.col(1) = std::sin(theta)*std::sin(phi)*R_rtp.col(0)
        + std::cos(theta)*std::sin(phi)*R_rtp.col(1)
        + std::cos(phi)*R_rtp.col(2);

    ret_mat.col(2) = std::cos(theta)*R_rtp.col(0) - std::sin(theta)*R_rtp.col(1);


    return ret_mat;


}