#include <Eigen/Core>
#include <iostream>

Eigen::MatrixXd build_MatrixXd(const double *data, const int rows, const int cols){
    /**
        * Given a data pointer and the dimensions, make
        * an Eigen MatrixXd object.
        * 
        * @param data The data pointer.
        * @param rows The number of rows.
        * @param cols The number of columns.
        * @return The Eigen::MatrixXd object.
    */

    // make space for the matrix
    Eigen::MatrixXd ret_mat(rows, cols);

    // fill it
    for (int i = 0; i < rows; ++i){

        for (int j = 0; j < cols; ++j){
            ret_mat(i, j) = data[i * cols + j];
        }
    }

    return ret_mat;

}


Eigen::MatrixXi build_MatrixXi(const int *data, const int rows, const int cols){
    /**
        * Given a data pointer and the dimensions, make
        * an Eigen MatrixXi object.
        * 
        * @param data The data pointer.
        * @param rows The number of rows.
        * @param cols The number of columns.
        * @return The Eigen::MatrixXi object.
    */

    // make space for the matrix
    Eigen::MatrixXi ret_mat(rows, cols);

    // fill it
    for (int i = 0; i < rows; ++i){

        for (int j = 0; j < cols; ++j){
            ret_mat(i, j) = data[i * cols + j];
        }
    }

    return ret_mat;
}

Eigen::VectorXd build_VectorXd(const double *data, const int rows){
    /**
        * Given a data pointer and the dimensions, make
        * an Eigen MatrixXd object.
        * 
        * @param data The data pointer.
        * @param rows The number of rows.
        * @return The Eigen::VectorXd object.
    */

    // make space for the matrix
    Eigen::VectorXd ret_vec(rows);

    // fill it
    for (int i = 0; i < rows; ++i){
        ret_vec(i) = data[i];
    }

    return ret_vec;


}

void to_c_array(const Eigen::MatrixXd &mat, double *data_ptr){
    /**
        * Given an Eigen::Matrix, get a c_pointer to the
        * data.
        * 
        * @param mat The Eigen::Matrix.
        * @param data_ptr The data pointer, should have been allocated first.
        * @return Nothing.
    */

    // the number of rows
    int rows = (int) mat.rows();
    int cols = (int) mat.cols();

    for (int i = 0; i < rows; ++i){

        for (int j = 0; j < cols; ++j){

            data_ptr[i*cols + j] = mat(i, j);
        }
    }

    return;
    
}