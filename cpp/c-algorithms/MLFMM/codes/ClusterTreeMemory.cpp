#ifndef MLFMM_CLUSTERTREEMEMORY_H_
#define MLFMM_CLUSTERTREEMEMORY_H_

/**
 *  \ingroup MeasurementTreeMemory
 *  \brief This struct keeps track of the measurement tree memory
 */


struct ClusterTreeMemory {


  ClusterTreeNode &get_root() {
    return (*memory_)[0];
  }

  std::vector<int>::size_type nsons(ClusterTreeNode *etn) {
    return etn->sons_.size();
  }

  void test(ClusterTreeNode etn,int i){
    ClusterTreeNode test = (*memory_)[etn.sons_[i]];
    //std::cout << "Test " << etn.sons_[i] <<std::endl;
  }

  ClusterTreeNode &son(ClusterTreeNode &etn,  std::vector<int>::size_type id) {//
    //std::cout << etn.sons_[id] << std::endl;
    return (*memory_)[etn.sons_[id]];
  }

  const ClusterTreeNode &son(const ClusterTreeNode &etn,
                             std::vector<int>::size_type id) const {
    //std::cout << "\tPos of son = " << etn.sons_[id] << std::endl;
    return (*memory_)[etn.sons_[id]];
  }

  //gives the cumulative sum of nodes of the tree up to level l
  int cumNumElements(int l) const {
    //root is lvl -1
    if (l == -1) return 1;
    //Maximum number of nodes is given by: \sum_{l=-1}^{L} 8^l = (8^{L+2}-1)/7
    else return (std::pow(8,l+2)-1)/7;
  }

  std::vector<ClusterTreeNode>::iterator lbegin(unsigned int level) {
    return (*memory_).begin() + levels_[level];
  }

  std::vector<ClusterTreeNode>::iterator lend(unsigned int level) {
    return (*memory_).begin() + levels_[level + 1];
  }

  ClusterTreeNode &get_element(std::vector<ClusterTreeNode>::size_type id) {
    return (*memory_)[id];
  }

  ClusterTreeNode *get_element_ptr(std::vector<ClusterTreeNode>::size_type id) {
    return &(*memory_)[id];
  }



  std::shared_ptr<std::vector<ClusterTreeNode>> memory_;
  int max_level_;

  std::vector<int> levels_;


};

#endif
