#include "ctools.h"
#include "evaluators.h"
#include <iostream>
#include <Eigen/Core>

using namespace Eigen;


void test_eigen(double test){

	std::cout << "Hey du Nudel!" << std::endl;

	Eigen::MatrixXd X(2, 2);

	X(0, 0) = 1.0;
	X(1, 0) = 2.0;
	X(0, 1) = 3.0;
	X(1, 1) = 4.0;

	std::cout << X << std::endl;

	double current = 1.0;

	Eigen::MatrixXd src(2, 6);
	
	src(0, 0) = 1.0;
	src(0, 1) = 1.0;
	src(0, 2) = 1.0;
	src(0, 3) = 3.0;
	src(0, 4) = 1.0;
	src(0, 5) = 1.0;

	src(1, 0) = 3.0;
	src(1, 1) = 1.0;
	src(1, 2) = 1.0;
	src(1, 3) = 5.0;
	src(1, 4) = 1.0;
	src(1, 5) = 1.0;

	std::cout << "src = " << src(0, 0) << std::endl;
	
	Eigen::MatrixXd tar(2, 3);
	
	tar(0, 0) = 0.0;
	tar(0, 1) = 0.0;
	tar(0, 2) = 0.0;

	tar(1, 0) = 0.1;
	tar(1, 1) = 0.1;
	tar(1, 2) = 0.1;	

	// Eigen::MatrixXd B = compute_B(src, tar, current);

	// std::cout << "B = " << B << std::endl;

	return;

}
