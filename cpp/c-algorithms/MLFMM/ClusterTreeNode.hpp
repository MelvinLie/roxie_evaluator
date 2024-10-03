#ifndef CLUSTERNODE_H_
#define CLUSTERNODE_H_

#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <Eigen/Sparse>

/**
 *  \ingroup MeasurementClusterNode
 *  \brief Cluster of measurements
 */

 // forward declaration of memory is necessary here
 struct ClusterTreeMemory;


 class ClusterTreeNode {

 public:
  //constructor
  ClusterTreeNode();

  void append_index(const int index);


  void set_memory(std::shared_ptr<ClusterTreeMemory> memory);

  void setup_cluster_box(Eigen::MatrixXd bbox);

  void set_center(Eigen::Vector3d in);

  void set_meas_table(std::vector<int> in);

  void make_meas_table();


   std::vector<int> indices_;
   std::vector<int> interaction_region_;
   std::vector<int> interaction_m2l_index_;
   std::vector<int> interaction_m2l_rot_index_;
   std::vector<int> near_field_;
   Eigen::VectorXi meas_table_; //(we could in future remove indices_ since it stores the same info)
   Eigen::MatrixXd bbox_;
   std::shared_ptr<ClusterTreeMemory> memory_;
   int father_;
   double diam_;
   std::vector<int> sons_;
   int pos_,level_;
   Eigen::Vector3d center_;

   //local FM expansion coefficients
   Eigen::MatrixXcd moments_;
   Eigen::MatrixXcd locals_;

   //matrices for the far field evaluation
   Eigen::SparseMatrix<std::complex<double>,Eigen::RowMajor>  local_mat_;

};

#endif
