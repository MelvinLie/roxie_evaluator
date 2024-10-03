#include "ClusterTreeNode.hpp"
 
ClusterTreeNode::ClusterTreeNode() { }

void ClusterTreeNode::append_index(const int index){
  indices_.push_back(index);
}


void ClusterTreeNode::set_memory(std::shared_ptr<ClusterTreeMemory> memory) {
  memory_ = memory;
  return;
}

void ClusterTreeNode::setup_cluster_box(Eigen::MatrixXd bbox) {
  bbox_ = bbox;
  center_ = bbox.colwise().mean();
  diam_ = (bbox.row(0) - bbox.row(1)).norm();
  return;
}

void ClusterTreeNode::set_center(Eigen::Vector3d in){
  center_ = in;
}

void ClusterTreeNode::set_meas_table(std::vector<int> in){

    meas_table_ = Eigen::Map<Eigen::VectorXi>(&(in.data()[0]),indices_.size());
}

void ClusterTreeNode::make_meas_table(){
    meas_table_ = Eigen::Map<Eigen::VectorXi>(&(indices_.data()[0]),indices_.size());
}
