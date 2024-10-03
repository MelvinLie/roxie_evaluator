#include "ClusterTree.hpp"

#include <Eigen/Dense>

ClusterTree::ClusterTree(){};

ClusterTree::ClusterTree(const Eigen::MatrixXd &pos, const Eigen::Matrix<double,2,3> &bounding_box) {

  //initialize the position matrix
  pos_ = pos;

  //initialize the bounding box
  bounding_box_ = bounding_box;

}

Eigen::MatrixXd ClusterTree::get_positions(){

  return pos_;

}

Eigen::Vector3d ClusterTree::get_positions_cog(){

  return pos_.colwise().mean();

}

Eigen::MatrixXd *ClusterTree::get_position_ptr(){

  return &pos_;

}

void ClusterTree::translate_positions(Eigen::Vector3d t){

  pos_.col(0) = pos_.col(0).array() + t(0);
  pos_.col(1) = pos_.col(1).array() + t(1);
  pos_.col(2) = pos_.col(2).array() + t(2);

}

void ClusterTree::scale_positions(double ratio){

  pos_ *= ratio;

}



std::vector<std::vector<int>> ClusterTree::bisect_index_list(Eigen::MatrixXd box, std::vector<int> index_list,std::vector<Eigen::MatrixXd> *bisection){

  double bound_marg = 1e-8;

  double x_sect[3] = {box(0,0)-bound_marg,0.5*(box(1,0)+box(0,0)),box(1,0)+bound_marg};
  double y_sect[3] = {box(0,1)-bound_marg,0.5*(box(1,1)+box(0,1)),box(1,1)+bound_marg};
  double z_sect[3] = {box(0,2)-bound_marg,0.5*(box(1,2)+box(0,2)),box(1,2)+bound_marg};

  std::vector<std::vector<int>> ret_list;

  //we separate into 8 subdomains
  ret_list.resize(8);
  bisection->resize(8);
  for (int i = 0; i < 8; ++i) (*bisection)[i].resize(2,3);

   //temporal index list with remaining indices
    std::vector<int> rem_indx_list;

    //index counter
    int current_index;

    //domain counter
    int l = 0;
    for (int i = 0; i < 2; ++i){

      for (int j = 0; j < 2; ++j){
        for(int k = 0; k < 2; ++k){

            (*bisection)[l](0,0) = x_sect[i];
            (*bisection)[l](1,0) = x_sect[i+1];
            (*bisection)[l](0,1) = y_sect[j];
            (*bisection)[l](1,1) = y_sect[j+1];
            (*bisection)[l](0,2) = z_sect[k];
            (*bisection)[l](1,2) = z_sect[k+1];

            for(int m = 0; m < index_list.size(); m++){
              current_index = index_list[m];

              if((x_sect[i] <= pos_(current_index,0)) & (pos_(current_index,0) < x_sect[i+1]) &
                 (y_sect[j] <= pos_(current_index,1)) & (pos_(current_index,1) < y_sect[j+1]) &
                 (z_sect[k] <= pos_(current_index,2)) & (pos_(current_index,2) < z_sect[k+1])){

                   //append index to index list
                   ret_list[l].push_back(current_index);

                 }
                 else{
                  //add index to the remaining ones
                  rem_indx_list.push_back(current_index);

                 }

            }
        //hand over remaining index list to avoid double indices
        index_list = rem_indx_list;
        rem_indx_list.clear();
        l++;
      }
    }
  }
  return ret_list;
}

void ClusterTree::init_cluster_tree(int max_level){


  //container for domain bisection
  std::vector<Eigen::MatrixXd> domain_bisection;

  //allocate MeasurementTreeMemory pointer
  cluster_tree_memory_ = std::make_shared<ClusterTreeMemory>();
  //set max level
  cluster_tree_memory_->max_level_ = max_level;
  //allocate a pointer to the tree nodes
  cluster_tree_memory_->memory_ = std::make_shared<std::vector<ClusterTreeNode>>();
  //allocate memory for the hole tree
  cluster_tree_memory_->memory_->resize(cluster_tree_memory_->cumNumElements(max_level));


  //get reference to the root node
  ClusterTreeNode &root = cluster_tree_memory_->get_root();
  //initialize the root node measurement indices
  for (int i = 0; i < pos_.rows() ; ++i) root.append_index(i);
  root.make_meas_table();
  //set memory pointer of root node
  root.set_memory(cluster_tree_memory_);
  //set root nodes center and bounding box
  root.setup_cluster_box(bounding_box_);
  //set level and position in memory of root node
  root.pos_ = 0;
  root.level_ = -1; //root is defined as level -1

  //setup level counter to -1
  lvl_ctr_ = -1;

  //setup the total node counter
  node_ctr_ = 1;

  //first level starts with root node
  cluster_tree_memory_->levels_.push_back(0);
  cluster_tree_memory_->levels_.push_back(1);


  //std::cout << "Generate Cluster Tree" << std::endl;
  generate_cluster_tree(0,1);

  cluster_tree_memory_->son(root,0);

  //sort level vector
  std::sort(cluster_tree_memory_->levels_.begin(),cluster_tree_memory_->levels_.end());
  //sort_positions();

}

void ClusterTree::print_cluster_tree(){

    ClusterTreeNode tmp_node;

    if((*cluster_tree_memory_->memory_).size() == 0){

      std::cout << "Initialize a cluster tree first!" << std::endl;
    }
    else{
        for (int n = 0; n < node_ctr_; ++n){
          tmp_node = (*cluster_tree_memory_->memory_)[n];
          std::cout << "------------------------------" << std::endl;
          std::cout << "Node (" << n << "):" << std::endl;
          std::cout << "\tpos =" << tmp_node.pos_ << std::endl;
          std::cout << "\tCenter = ( " << tmp_node.center_(0) << " , ";
          std::cout  << tmp_node.center_(1) << " , " << tmp_node.center_(2) << " )" << std::endl;
          std::cout << "\tDiam = " << tmp_node.diam_ << std::endl;
          std::cout << "\tBbox = ( " << tmp_node.bbox_(0,0) << " , ";
          std::cout  << tmp_node.bbox_(0,1) << " , " << tmp_node.bbox_(0,2) << " )" << std::endl;
          std::cout << "\t       ( " << tmp_node.bbox_(1,0) << " , ";
          std::cout  << tmp_node.bbox_(1,1) << " , " << tmp_node.bbox_(1,2) << " )" << std::endl;
          std::cout << "\tNumber of entities = " << tmp_node.indices_.size() << std::endl;
          std::cout << "\tEntities table = " << tmp_node.meas_table_.transpose() << std::endl;
          std::cout << "\tEntities indices = ";
          for (int i = 0; i < tmp_node.indices_.size(); ++i) std::cout << tmp_node.indices_[i] << " ";
          std::cout << std::endl;
          std::cout << "\tLevel = " << tmp_node.level_+1 << std::endl;
          std::cout << "\tNumber of sons = " << tmp_node.sons_.size() << std::endl;
          std::cout << "\tsons = ";
          for (int i = 0 ; i <  tmp_node.sons_.size() ; ++i) std::cout << tmp_node.sons_[i] << " , ";
          std::cout << std::endl;
        }

    }
}


std::shared_ptr<ClusterTreeMemory> ClusterTree::get_tree_memory(){

  return cluster_tree_memory_;
}

void ClusterTree::set_bounding_box(const Eigen::Matrix<double,2,3> &bbox){
    bounding_box_ = bbox;
}

void ClusterTree::set_min_numel(int min_numel){

    min_numel_ = min_numel;

}

void ClusterTree::generate_cluster_tree(int mem_from,int mem_to){

      //increment level counter
      lvl_ctr_ += 1;
      //number of current root nodes
      int num_root = mem_to - mem_from;
      //position of first son in memory = mem_to;
      //initialize son counter
      int son_ctr = 0;
      //bisected indices list
      std::vector<std::vector<int>> bs_idx_list;
      //container for domain bisection
      std::vector<Eigen::MatrixXd> domain_bisection;

      //iterate over all given nodes
      for (int n = 0; n < num_root; ++n){

        if((lvl_ctr_ < min_tree_level_) || (*cluster_tree_memory_->memory_)[mem_from + n].indices_.size() > min_numel_){

          //number of measurements in this node
          //int num_node_meas = (*cluster_tree_memory_->memory_)[mem_from + n].indices_.size();

          //bisect the domain and separate the index list of this node
          bs_idx_list = bisect_index_list((*cluster_tree_memory_->memory_)[mem_from + n].bbox_,
                                            (*cluster_tree_memory_->memory_)[mem_from + n].indices_,
                                            &domain_bisection);

          for (int s = 0; s < bs_idx_list.size(); ++s){
            //it can be that there are no measurements in this subsection!
            if (bs_idx_list[s].size() == 0) continue;
            else{
              //there are some measurements here, this is a node
              //increment the total node counter
              node_ctr_++;
              //initialize the new node
              (*cluster_tree_memory_->memory_)[mem_to + son_ctr].indices_ = bs_idx_list[s];
              //initialize the measurement table (we could in future remove indices_ since it stores the same info)
              (*cluster_tree_memory_->memory_)[mem_to + son_ctr].set_meas_table(bs_idx_list[s]);
              //give it access to memory
              (*cluster_tree_memory_->memory_)[mem_to + son_ctr].set_memory(cluster_tree_memory_);
              //setup his father
              (*cluster_tree_memory_->memory_)[mem_to + son_ctr].father_ = mem_from + n;
              //this the geometric information
              (*cluster_tree_memory_->memory_)[mem_to + son_ctr].setup_cluster_box(domain_bisection[s]);
              //mark level in level indicator
              (*cluster_tree_memory_->memory_)[mem_to + son_ctr].level_ = lvl_ctr_;
              //position of this node in memory_
              (*cluster_tree_memory_->memory_)[mem_to + son_ctr].pos_ = node_ctr_-1;
              //add this position for the son to father
              (*cluster_tree_memory_->memory_)[mem_from + n].sons_.push_back(mem_to + son_ctr);
              //increment son counter
              son_ctr++;
            }

          }

        }
  }
  if (lvl_ctr_ < cluster_tree_memory_->max_level_){
    //refine further
    generate_cluster_tree(mem_to, mem_to + son_ctr);

  }
  //save a reference to the first element of next level in levels_
  cluster_tree_memory_->levels_.push_back(mem_to + son_ctr);

}
