#include <Eigen/Dense>
#include <Eigen/Core>
#include <iostream>

#include "mesh_tools.h"

#define PI 3.141592653589793238462643383279

double compute_self_inductance_cylinder(const double length,
											const double radius) {

	/**
	* Compute the self inductance of a cylindrical current line.
	* 
	* @param length The length of the conductor.
	* @param radius The radius of the conductor.
	* @return The self inductance.
	*/
	return 1e-7*(2.0 * length * (std::log(2.0 * length / radius) - 1)
				+ 8.0 * radius / PI
				- radius * radius / length);

}

double compute_self_inductance_adjacent_straight_elements(const Eigen::Vector3d &d1,
															const Eigen::Vector3d &d2,
															const double radius) {
/**
* Compute the self inductance of a cylindrical current line.
*
* @param seg1 The first segment defined as a tuple of points.
* @param seg2 The second segment defined as a tuple of points.
* @param radius The radius of the conductors
* @return The self inductance.
*/
		
	// the lengths of the elements
	double l1 = d1.norm();
	double l2 = d2.norm();

	// the projection of d1 on d2
	double d1_d2 = d1.dot(d2);

	// the angle between them

	double alpha;

	// this is to avoid numerical instabilities
	if ((d1_d2 / l1 / l2) > 1.0) {
		alpha = 0.0;
	}
	else if ((d1_d2 / l1 / l2) < -1.0) {
		alpha = PI;
	}
	else {
		alpha = std::acos(d1_d2 / l1 / l2);
	}
		
	// auxiliary variables
	double sa = std::sin(alpha);
	double ca = std::cos(alpha);
	double arg1 = (l2 + l1 * ca) / l1 / sa;
	double arg2 = (l1 + l2 * ca) / l2 / sa;
	double arg3 = ca / sa;
	double arg4 = (1 - ca) / sa;
	double b = radius / 2.0;
	double L;

	// compute the self inductance
	L = 2e-7 * ca * (l1 * std::asinh(arg1)
		+ l2 * std::asinh(arg2)
		- (l1 + l2)*std::asinh(arg3)
		- 2.0 * b / std::sqrt(1.0 - std::cos(alpha))*std::asinh(arg4));

	return L;
}

double inv_dist(const double xi1,
				const double xi2,
				const Eigen::Vector3d &p11,
				const Eigen::Vector3d &p12,
				const Eigen::Vector3d &p21,
				const Eigen::Vector3d &p22) {
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

	Eigen::Vector3d r1 = 0.5*(p12*(xi1 + 1) - p11 * (xi1 - 1));
	Eigen::Vector3d r2 = 0.5*(p22*(xi2 + 1) - p21 * (xi2 - 1));

	double dist = (r2 - r1).norm();

	return 1.0 / dist;

}

double compute_interaction(const Eigen::MatrixXd &Q,
							const Eigen::Vector3d &p11,
							const Eigen::Vector3d &p12,
							const Eigen::Vector3d &p21,
							const Eigen::Vector3d &p22) {

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

	// the number of integration points
	int num_pts = (int) Q.rows();

	// compute the difference vectors
	Eigen::Vector3d d1 = p12 - p11;
	Eigen::Vector3d d2 = p22 - p21;

	// the projection
	double d1_d2 = d1.dot(d2);

	// we could avoid the integration for orthogonal elements?

	// integrate
	double int_val = 0.0;


	for (int i = 0; i < num_pts; ++i) {

		for (int j = 0; j < num_pts; ++j) {
				
			int_val += Q(i, 1) * Q(j, 1) * inv_dist(Q(i, 0),
													Q(j, 0),
													p11,
													p12,
													p21,
													p22);

		}
	}


	return d1_d2*int_val;
}

double compute_self_inductance(const Eigen::MatrixXd &segments,
								const double radius,
								const int num_points,
								const int is_open) {

	// the number of segments
	int num_seg = (int) segments.rows();
		
	// get the diameter of the domain
	double diam = compute_diam_segments(segments);

	// containers for difference vectors
	Eigen::Vector3d d1, d2;

	// container variable for the length
	double length;

	// the value of the mutual inductance
	double L = 0.0;

	#pragma omp parallel
	{

		// the result of this process
		double my_L = 0.0;

		// container for the quadrature points
		Eigen::MatrixXd Qnear = get_1D_gauss_integration_points(num_points);
		Eigen::MatrixXd Qfar = get_1D_gauss_integration_points(2);

		// loop over the segments(non identical)
		for (int i = 0; i < num_seg; ++i) {

			#pragma omp single nowait
			{
				for (int j = 0; j < num_seg; ++j) {

					if (std::abs(i - j) > 1) {
							

						// the distance between the segments
						double dist = 1.0 / inv_dist(0.0, 0.0,
													segments.block(i, 0, 1, 3).transpose(),
													segments.block(i, 3, 1, 3).transpose(),
													segments.block(j, 0, 1, 3).transpose(),
													segments.block(j, 3, 1, 3).transpose());


						if (dist < 0.05*diam){
							// no problem Gaussian quadrature
							my_L += compute_interaction(Qnear,
								segments.block(i, 0, 1, 3).transpose(),
								segments.block(i, 3, 1, 3).transpose(),
								segments.block(j, 0, 1, 3).transpose(),
								segments.block(j, 3, 1, 3).transpose());
						}
						else{
							// no problem Gaussian quadrature
							my_L += compute_interaction(Qfar,
								segments.block(i, 0, 1, 3).transpose(),
								segments.block(i, 3, 1, 3).transpose(),
								segments.block(j, 0, 1, 3).transpose(),
								segments.block(j, 3, 1, 3).transpose());
						}

					}

				}
			}
		}

		#pragma omp critical
		{
			// this factor is missing in the computation of the interaction
			L += 0.25e-7*my_L;
		}

	}

	// loop over the identical segments
	for (int i = 0; i < num_seg; ++i) {
	
		length = (segments.block(i, 0, 1, 3) 
						- segments.block(i, 3, 1, 3)).norm();
			
		L += compute_self_inductance_cylinder(length, radius);

		if(length < 20*radius){

			std::cout << "Warning! The segment length is in the range of the strand length!" << std::endl;
			std::cout << "Consider decreasing the discretization!" << std::endl; 
		}
		
	}


	// loop over the corners
	for (int i = 0; i < num_seg; ++i) {

		if ((i == 0) && (is_open == 1)) {

			d1 = segments.block(num_seg-1, 3, 1, 3).transpose() \
				- segments.block(num_seg-1, 0, 1, 3).transpose();

			d2 = segments.block(i, 3, 1, 3).transpose() \
				- segments.block(i, 0, 1, 3).transpose();

		}
		else if (i > 0) {

			d1 = segments.block(i - 1, 3, 1, 3).transpose() \
				- segments.block(i - 1, 0, 1, 3).transpose();

			d2 = segments.block(i, 3, 1, 3).transpose() \
				- segments.block(i, 0, 1, 3).transpose();
		}
			
		L += compute_self_inductance_adjacent_straight_elements(d1, d2, radius);
	}


	return L;


}
