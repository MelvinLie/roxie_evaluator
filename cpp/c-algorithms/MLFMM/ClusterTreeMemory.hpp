#ifndef CLUSTERTREEMEMORY_H_
#define CLUSTERTREEMEMORY_H_

#include <iostream>
#include <fstream>
#include "ClusterTreeNode.hpp"

/**
 *  \ingroup ClusterTreeMemory
 *  \brief This struct keeps track of the measurement tree memory
 */

struct ClusterTreeMemory {


  ClusterTreeNode &get_root();

  std::vector<int>::size_type nsons(ClusterTreeNode *etn);

  void test(ClusterTreeNode etn,int i);

  ClusterTreeNode &son(ClusterTreeNode &etn,  std::vector<int>::size_type id);

  const ClusterTreeNode &son(const ClusterTreeNode &etn,
                             std::vector<int>::size_type id) const;

  //gives the cumulative sum of nodes of the tree up to level l
  int cumNumElements(int l) const;

  std::vector<ClusterTreeNode>::iterator lbegin(unsigned int level);

  std::vector<ClusterTreeNode>::iterator lend(unsigned int level);

  ClusterTreeNode &get_element(std::vector<ClusterTreeNode>::size_type id);

  ClusterTreeNode *get_element_ptr(std::vector<ClusterTreeNode>::size_type id);

  std::shared_ptr<std::vector<ClusterTreeNode>> memory_;
  int max_level_;
  std::vector<int> levels_;

};

#endif
