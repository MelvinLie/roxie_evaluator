#ifndef UTILS_H_
#define UTILS_H_

#include <Eigen/Dense>
#include <Eigen/Core>

/**
  * Compute the diameter based on all the sources there are.

  * @param src The source matrix.
  * @return The diameter.
*/
double compute_diam_src(const Eigen::MatrixXd &src);

/**
  * Compute the diameter based on all the segments there are.

  * @param segments The segments matrix.
  * @return The diameter.
*/
double compute_diam_segments(const Eigen::MatrixXd &segments);

/**
  * Evaluate a point in a polygon based on the local
  coordinates (u, v).

  * @param u The local coordinate between p1 and p2.
  * @param v The local coordinate between p1 and p4.
  * @param p1 The coordinates of the point p1.
  * @param p2 The coordinates of the point p2.
  * @param p3 The coordinates of the point p3.
  * @param p4 The coordinates of the point p4.
  * @return The vector in 3D.
*/
Eigen::Vector3d eval_polygon(const double u, const double v, const Eigen::Vector3d &p1,
                                                          const Eigen::Vector3d &p2,
                                                          const Eigen::Vector3d &p3,
                                                          const Eigen::Vector3d &p4 );

/**
  * Get the 1D Gaussian integration points.

  * @param num_points number of integration points.
  * @return A matrix with the integration points.
*/
Eigen::MatrixXd get_1D_gauss_integration_points(const int num_points);

#endif
