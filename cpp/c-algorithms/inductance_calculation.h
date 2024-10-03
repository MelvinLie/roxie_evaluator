#ifndef INDUCTANCE_CALC_H_
#define INDUCTANCE_CALC_H_

/*****************************************
This code provides functions needed to compute the mutual and
self inductances of filament currents.

Autor: Melvin Liebsch
email: melvin.liebsch@cern.ch
*****************************************/


/**
* Compute the self inductance of a cylindrical current line.
* 
* @param length The length of the conductor.
* @param radius The radius of the conductor.
* @return The self inductance.
*/
double compute_self_inductance_cylinder(const double length, const double radius);

/**
* Compute the self inductance of a cylindrical current line.
*
* @param seg1 The first segment defined as a tuple of points.
* @param seg2 The second segment defined as a tuple of points.
* @param radius The radius of the conductors
* @return The self inductance.
*/
double compute_self_inductance_adjacent_straight_elements(const Eigen::Vector3d &d1,
														  const Eigen::Vector3d &d2,
														  const double radius);

/**
* Evaluate the inverse distance between the segments seg1
* and seg2 for the parameters xi1 and xi2.
*
* @param xi1 The parameter between - 1 and 1 to evaluate segment 1.
* @param xi2 The parameter between - 1 and 1 to evaluate segment 2.
* @param p11 The point specifying the beginning of segment 1.
* @param p12 The point specifying the end of segment 1.
* @param p21 The point specifying the beginning of segment 1.
* @param p22 The point specifying the end of segment 1.
* @return The inverse distance.
*/
double inv_dist(const double xi1,
					const double xi2,
					const Eigen::Vector3d &p11,
					const Eigen::Vector3d &p12,
					const Eigen::Vector3d &p21,
					const Eigen::Vector3d &p22);

/**
* Compute the interaction between two segments by Gaussian integration.
* @param Q The quadrature points and weights, as they are
* generated from the function get_1D_gauss_integration_points in Utils.
* @param p11 The point specifying the beginning of segment 1.
* @param p12 The point specifying the end of segment 1.
* @param p21 The point specifying the beginning of segment 1.
* @param p22 The point specifying the end of segment 1.
* @return The result of the Gaussian integration
*/
double compute_interaction(const Eigen::MatrixXd &Q,
								const Eigen::Vector3d &p11,
								const Eigen::Vector3d &p12,
								const Eigen::Vector3d &p21,
								const Eigen::Vector3d &p22);

/**
* Compute the self inductance of a filament current
* loop, which is represented by a list of straigh line segments.
* @param segments A matrix with M rows and 6 columns. Each row 
* specifies the start and end points of a segment.
* @param radius The radius of the filament currents.
* @param num_points The number of points for the Gaussian integration.
* @param is_open A flag specifying if the current loop is fully closed or not.
* @return The self inductance.
*/
double compute_self_inductance(const Eigen::MatrixXd &segments,
									const double radius,
									const int num_points,
									const int is_open);


#endif