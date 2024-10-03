#ifndef MLFMM_CLUSTERTREE_H_
#define MLFMM_CLUSTERTREE_H_

#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include "ClusterTreeMemory.hpp"



 class ClusterTree
 {

 public:
  //constructors
  ClusterTree();
  ClusterTree(const Eigen::MatrixXd &pos, const Eigen::Matrix<double,2,3> &bounding_box);

  // public member functions
  Eigen::MatrixXd get_positions();

  Eigen::Vector3d get_positions_cog();

  Eigen::MatrixXd *get_position_ptr();

  void translate_positions(Eigen::Vector3d t);

  void scale_positions(double ratio);

  std::vector<std::vector<int>> bisect_index_list(Eigen::MatrixXd box, std::vector<int> index_list,std::vector<Eigen::MatrixXd> *bisection);

  void init_cluster_tree(int max_level);

  void print_cluster_tree();

  std::shared_ptr<ClusterTreeMemory> get_tree_memory();

  void set_bounding_box(const Eigen::Matrix<double,2,3> &bbox);

  void set_min_numel(int min_numel);


 private:
   // private member variables (end with bar _)
   Eigen::MatrixXd pos_;
   Eigen::MatrixXd bounding_box_;
   std::shared_ptr<ClusterTreeMemory> cluster_tree_memory_;
   //to count the level in generate_cluster_tree
   int lvl_ctr_;
   //to count the current number of nodes in generate_cluster_tree
   int node_ctr_;
   //minimum number of measurements in a cell with sons
   int min_numel_ = 1;
   int min_tree_level_ = 2;

   // private member functions
   void generate_cluster_tree(int mem_from, int mem_to);


 };


#endif
