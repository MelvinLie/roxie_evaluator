#include <Eigen/Dense>
#include <Eigen/Core>

double compute_diam_src(const Eigen::MatrixXd &src) {
/**
  * Compute the diameter based on all the sources there are.

  * @param src The source matrix.
  * @return The diameter.
*/

  double max_x = src.col(0).array().maxCoeff();
  double max_y = src.col(1).array().maxCoeff();
  double max_z = src.col(2).array().maxCoeff();

  double min_x = src.col(0).array().minCoeff();
  double min_y = src.col(1).array().minCoeff();
  double min_z = src.col(2).array().minCoeff();

  for (int i = 1 ; i <= 3 ; ++i){

    double this_max_x = src.col(3*i+0).array().maxCoeff();
    double this_max_y = src.col(3*i+1).array().maxCoeff();
    double this_max_z = src.col(3*i+2).array().maxCoeff();

    double this_min_x = src.col(3*i+0).array().minCoeff();
    double this_min_y = src.col(3*i+1).array().minCoeff();
    double this_min_z = src.col(3*i+2).array().minCoeff();

    if(max_x < this_max_x) max_x = this_max_x;
    if(max_y < this_max_y) max_y = this_max_y;
    if(max_z < this_max_z) max_z = this_max_z;

    if(min_x > this_min_x) min_x = this_min_x;
    if(min_y > this_min_y) min_y = this_min_y;
    if(min_z > this_min_z) min_z = this_min_z;

  }

  Eigen::Vector3d diams;
  diams(0) = max_x - min_x;
  diams(1) = max_y - min_y;
  diams(2) = max_z - min_z;

  return diams.array().maxCoeff();

}

double compute_diam_segments(const Eigen::MatrixXd &segments){
/**
  * Compute the diameter based on all the segments there are.

  * @param segments The segments matrix.
  * @return The diameter.
*/

  double max_x = segments.col(0).array().maxCoeff();
  double max_y = segments.col(1).array().maxCoeff();
  double max_z = segments.col(2).array().maxCoeff();

  double min_x = segments.col(0).array().minCoeff();
  double min_y = segments.col(1).array().minCoeff();
  double min_z = segments.col(2).array().minCoeff();

  Eigen::Vector3d diams;
  diams(0) = max_x - min_x;
  diams(1) = max_y - min_y;
  diams(2) = max_z - min_z;

  return diams.array().maxCoeff();

}


Eigen::Vector3d eval_polygon(const double u, const double v, const Eigen::Vector3d &p1,
                                                          const Eigen::Vector3d &p2,
                                                          const Eigen::Vector3d &p3,
                                                          const Eigen::Vector3d &p4 ){

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

    // u is the coordinate in N direction
    // v is the coordinate in B direction

  	//the return value
    Eigen::Vector3d  ret_val;

    //compute the midpoint
    ret_val(0) = p1(0,0)*(1-u)*(1-v) \
                + p2(0,0)*u*(1-v) \
                + p3(0,0)*u*v \
                + p4(0,0)*(1-u)*v;

    //compute the midpoint
    ret_val(1) = p1(1,0)*(1-u)*(1-v) \
                + p2(1,0)*u*(1-v) \
                + p3(1,0)*u*v \
                + p4(1,0)*(1-u)*v;

    //compute the midpoint
    ret_val(2) = p1(2,0)*(1-u)*(1-v) \
                + p2(2,0)*u*(1-v) \
                + p3(2,0)*u*v \
                + p4(2,0)*(1-u)*v;

    

    return ret_val;

}

Eigen::MatrixXd get_1D_gauss_integration_points(const int num_points){
/**
  * Get the 1D Gaussian integration points.

  * @param num_points number of integration points.
  * @return A matrix with the integration points.
*/

  //these points have been generated with
  //https://keisan.casio.com/exec/system/1329114617
  
  //we return a matrix of size (num_points x 2) where the first column 
  //stores the points and the second row stores the weights
  Eigen::MatrixXd ret_val(num_points,2);

  
  switch(num_points){

    case 1:
      ret_val << 0. ,  2.;
      break;
    case 2:
      ret_val << -0.5773502691896257645092,	1.,
                  0.5773502691896257645092,	1.;
      break;
    case 3:
      ret_val << -0.7745966692414833770359,	0.5555555555555555555556,
                  0.,	0.8888888888888888888889,
                  0.7745966692414833770359,	0.555555555555555555556;
      break;
    case 4:
      ret_val << -0.861136311594052575224,	0.3478548451374538573731,
                -0.3399810435848562648027,	0.6521451548625461426269,
                0.3399810435848562648027,	0.6521451548625461426269,
                0.861136311594052575224,	0.3478548451374538573731;
      break;
    case 5:
      ret_val << -0.9061798459386639927976,	0.2369268850561890875143,
                -0.5384693101056830910363,	0.4786286704993664680413,
                0.,	0.5688888888888888888889,
                0.5384693101056830910363,	0.4786286704993664680413,
                0.9061798459386639927976,	0.2369268850561890875143;
      break;
    case 6:
      ret_val << -0.9324695142031520278123,	0.1713244923791703450403,
                -0.661209386466264513661,	0.3607615730481386075698,
                -0.2386191860831969086305,	0.4679139345726910473899,
                0.238619186083196908631,	0.46791393457269104739,
                0.661209386466264513661,	0.3607615730481386075698,
                0.9324695142031520278123,	0.1713244923791703450403;
      break;
    case 7:
      ret_val << -0.9491079123427585245262,	0.1294849661688696932706,
                -0.7415311855993944398639, 0.2797053914892766679015,
                -0.4058451513773971669066,	0.38183005050511894495,
                0,	0.417959183673469387755,
                0.4058451513773971669066,	0.38183005050511894495,
                0.7415311855993944398639,	0.279705391489276667901,
                0.9491079123427585245262,	0.129484966168869693271;
      break;
    case 20:
      ret_val << -0.9931285991850949247861,	0.0176140071391521183119,
                -0.9639719272779137912677,	0.04060142980038694133104,
                -0.9122344282513259058678,	0.0626720483341090635695,
                -0.8391169718222188233945,	0.0832767415767047487248,
                -0.7463319064601507926143,	0.1019301198172404350368,
                -0.6360536807265150254528,	0.1181945319615184173124,
                -0.5108670019508270980044,	0.1316886384491766268985,
                -0.3737060887154195606725,	0.1420961093183820513293,
                -0.2277858511416450780805,	0.1491729864726037467878,
                -0.07652652113349733375464,	0.1527533871307258506981,
                0.0765265211334973337546,	0.152753387130725850698,
                0.2277858511416450780805,	0.149172986472603746788,
                0.3737060887154195606725,	0.142096109318382051329,
                0.5108670019508270980044,	0.1316886384491766268985,
                0.6360536807265150254528,	0.1181945319615184173124,
                0.7463319064601507926143,	0.101930119817240435037,
                0.8391169718222188233945,	0.083276741576704748725,
                0.9122344282513259058678,	0.0626720483341090635695,
                0.9639719272779137912677,	0.040601429800386941331,
                0.9931285991850949247861,	0.0176140071391521183119;
      break;
    //we stop here. feel free to add higher order integration rules (or increase the number of digits) from https://keisan.casio.com/exec/system/1329114617 if neccessary.
  }

  return ret_val;

}
