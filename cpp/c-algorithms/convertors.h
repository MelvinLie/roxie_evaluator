#ifndef CONVERTORS_H
#define CONVERTORS_H

#include <Eigen/Core>

/**
    * Given a data pointer and the dimensions, make
    * an Eigen MatrixXd object.
    * 
    * @param data The data pointer.
    * @param rows The number of rows.
    * @param cols The number of columns.
    * @return The Eigen::MatrixXd object.
*/
Eigen::MatrixXd build_MatrixXd(const double *data, const int rows, const int cols);

/**
    * Given a data pointer and the dimensions, make
    * an Eigen MatrixXi object.
    * 
    * @param data The data pointer.
    * @param rows The number of rows.
    * @param cols The number of columns.
    * @return The Eigen::MatrixXi object.
*/
Eigen::MatrixXi build_MatrixXi(const int *data, const int rows, const int cols);

/**
    * Given a data pointer and the dimension, make
    * an Eigen VectorXd object.
    * 
    * @param data The data pointer.
    * @param rows The number of rows.
    * @return The Eigen::MatrixXd object.
*/
Eigen::VectorXd build_VectorXd(const double *data, const int rows);

/**
    * Given an Eigen::Matrix, get a c_pointer to the
    * data.
    * 
    * @param mat The Eigen::Matrix.
    * @return A pointer to the data.
*/
void to_c_array(const Eigen::MatrixXd &mat, double *data_ptr);

#endif