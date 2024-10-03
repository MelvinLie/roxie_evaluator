#ifndef MLFMM_MLFMM_H_
#define MLFMM_MLFMM_H_

#include <Eigen/Dense>
#include <Eigen/Sparse>


template <typename Monitor>
 class Mlfmm {

 public:
  //constructors

  Mlfmm(const Eigen::MatrixXd &src,const Eigen::MatrixXd &src_vec, const Eigen::MatrixXd &tar) {

    //get the positions
    src_ = src;

    //get the source vectors
    src_vec_ = src_vec;

    //get the targets
    tar_ = tar;

    //compute the bounding box
    bbox_ = compute_bounding_box(src_, tar_);

  }

  void initialize(int max_tree_lvl = 3,int L = 9, int min_numel = 1,bool print=false){


    //make the field monitor (vector potential, flux density or also a measurement device)
    Monitor monitor_;

    //get the dimension of the output space
    dim_output_space_ = monitor_.get_output_space_dimension();


    //number of targets
    num_tar_ = tar_.rows();

    //set min numel
    min_numel_ = min_numel;
    //harmonics order
    L_ = L;
    //set the number of coefficients
    num_coeffs_ = (L_+1)*(L_+1);
    //set the maximum tree level
    max_tree_lvl_ = max_tree_lvl;
    //construct the cluster trees
    src_tree_ = ClusterTree(src_,bbox_);
    tar_tree_ = ClusterTree(tar_,bbox_);
    //setup the cluster trees
    src_tree_.set_min_numel(min_numel);//
    tar_tree_.set_min_numel(min_numel);
    //initialize them
    src_tree_.init_cluster_tree(max_tree_lvl_);
    tar_tree_.init_cluster_tree(max_tree_lvl_);

    if(print){
      src_tree_.print_cluster_tree();
      tar_tree_.print_cluster_tree();
    }

    //compute moment matrices
    m2m_matrices_.fill(L_, max_tree_lvl_, bbox_, MultipoleMomentContainer::SOURCE);
    //compute local matrices
    l2l_matrices_.fill(L_, max_tree_lvl_, bbox_, MultipoleMomentContainer::LOCAL);


    //std::cout << "number fo multipoles = " << L_ << std::endl;

    //to have the box diameter
    compute_box_diam(bbox_);

    //make the near field and interaction region
    make_ir_and_nf();

    //make the leafs local matrices
    make_leafs_local_matrices();

  }

  Eigen::MatrixXd compute_A(int max_tree_lvl = 3,int L = 9, int min_numel = 1){


    //std::cout << "initialize" << std::endl;

    //initialize
    initialize(max_tree_lvl,L,min_numel,false);

    //allocate space for A
    Eigen::MatrixXd A(num_tar_,3);
    A.setZero();

    //std::cout << "compute_leaf_moments" << std::endl;

    //compute the leaf moments
    compute_leaf_moments();

    //std::cout << "upward_pass" << std::endl;
    //upward pass
    upward_pass();

    //std::cout << "downward_pass" << std::endl;

    //downward pass
    downward_pass();

    //std::cout << "evaluate" << std::endl;

    //std::cout << "Max tree level = " << max_tree_lvl_ << std::endl;

    Eigen::MatrixXd A_nf = evaluate_near_field();
    Eigen::MatrixXd A_ff = evaluate_far_field();

    //std::cout << "A_ff" << std::endl;
    //std::cout << A_ff.block(0,0,4,3) << std::endl;

    //std::cout << "A_nf" << std::endl;
    //std::cout << A_nf << std::endl;
    //compute near field
    A += A_ff;

    //compute far field
    A +=  A_nf;

    return A;
  }

  void set_bounding_box(const Eigen::Matrix<double,2,3> &bbox){

    bbox_ = bbox;
  }

  Eigen::MatrixXd compute_all_interactions(){

    //return matrix
    Eigen::MatrixXd ret_mat(num_tar_,dim_output_space_);
    ret_mat.setZero();

    #pragma omp parallel
    {

      //this is the result of this core
      Eigen::MatrixXd my_result(num_tar_,dim_output_space_);
      my_result.setZero();

      #pragma omp for
      for(int i = 0; i < num_tar_; ++i){
        //target loop

        for(int j = 0; j < src_.rows(); ++j){

          //source loop
          my_result.row(i) += monitor_.evaluate_near_field_interaction(tar_.row(i),src_.row(j),src_vec_.row(j));

        }



    }
    #pragma omp critical
    {
      ret_mat += my_result;
    }

  }

  return 1e-7*ret_mat;


  }

  Eigen::MatrixXd compute_all_interactions_serial(){

    //return matrix
    Eigen::MatrixXd ret_mat(num_tar_,dim_output_space_);
    ret_mat.setZero();

    for(int i = 0; i < num_tar_; ++i){
      //target loop

      for(int j = 0; j < src_.rows(); ++j){

        //source loop
        ret_mat.row(i) += monitor_.evaluate_near_field_interaction(tar_.row(i),src_.row(j),src_vec_.row(j));

        }
    }


  return 1e-7*ret_mat;

  }

private:

  //------------------------------------------------------------
  // Private member variables
  //------------------------------------------------------------
  //source points
  Eigen::MatrixXd src_;
  //source vectors
  Eigen::MatrixXd src_vec_;
  //target points
  Eigen::MatrixXd tar_;
  //bounding box
  Eigen::Matrix<double,2,3> bbox_;
  //cluster tree for the sources
  ClusterTree src_tree_;
  //cluster tree for the targets
  ClusterTree tar_tree_;
  //minimum number of entities in one cell
  int min_numel_ = 1;
  //number of targets
  int num_tar_;
  //output space dimension
  int dim_output_space_;
  //number of fmm components
  //int num_comp_ = 3;
  //max harmonics order
  int L_ = 9;
  //number of fmm coefficients
  int num_coeffs_ = (L_+1)*(L_+1);
  //we count the number of near field interactions
  int num_near_field_interactions_;

  //container for m2m matrices
  MultipoleMomentContainer m2m_matrices_;
  //container for l2l matrices
  MultipoleMomentContainer l2l_matrices_;

  //maximum tree level
  int max_tree_lvl_ = 3;


  //container for m2l matrices. These are a unique lists. The measurement and element trees
  //store the indices for the transformation matrix in this list
  std::vector<Eigen::SparseMatrix<std::complex<double>,Eigen::RowMajor>,
              Eigen::aligned_allocator<Eigen::SparseMatrix<std::complex<double>,Eigen::RowMajor>>> m2l_matrices_;

  std::vector<Eigen::SparseMatrix<std::complex<double>,Eigen::RowMajor>,
              Eigen::aligned_allocator<Eigen::SparseMatrix<std::complex<double>,Eigen::RowMajor>>> m2l_tansl_;

  std::vector<Eigen::SparseMatrix<std::complex<double>,Eigen::RowMajor>,
              Eigen::aligned_allocator<Eigen::SparseMatrix<std::complex<double>,Eigen::RowMajor>>> m2l_rot_;

  //near field interaction list (el_indx , meas_indx)
  std::vector<std::vector<int>> near_field_interaction_list_;

  //difference vectors corresponding to the m2l transfer in the unique list
  std::vector<Eigen::Vector3d,Eigen::aligned_allocator<Eigen::Vector3d>> d_vecs_;
  //difference vectors corresponding to the m2l rotations in the unique list
  std::vector<Eigen::Vector3d,Eigen::aligned_allocator<Eigen::Vector3d>> d_rot_vecs_;

  //cell size
  double hx_,hy_,hz_;

  //diagonal of box
  Eigen::Vector3d diag_;

  //distance from center to corner of box for level 0
  double diam_;

  //indices of leafs
  std::vector<int> local_leafs_;
  std::vector<int> source_leafs_;


  //The field monitor operator (Magnetic vector potential or B field or something else)
  Monitor monitor_;


  //------------------------------------------------------------
  // Private functions
  //------------------------------------------------------------

  Eigen::Matrix<double,2,3> compute_bounding_box(const Eigen::MatrixXd &src, const Eigen::MatrixXd &pnt){

      //compute center positions
      Eigen::MatrixXd all_points(src.rows()+pnt.rows(),3);
      all_points.block(0,0,src.rows(),3) = src.block(0,0,src.rows(),3);
      all_points.block(src.rows(),0,pnt.rows(),3) = pnt.block(0,0,pnt.rows(),3);

      //min max coordinates
      Eigen::Vector3d min_coeffs = all_points.colwise().minCoeff();
      Eigen::Vector3d max_coeffs = all_points.colwise().maxCoeff();

      //std::cout << "min = " << min_coeffs.transpose() << std::endl;
      //std::cout << "max = " << max_coeffs.transpose() << std::endl;


      Eigen::Vector3d axes_widths = max_coeffs-min_coeffs;

      //std::cout << "axes_widths = " << axes_widths.transpose() << std::endl;

      //allocate the bounding box
      Eigen::Matrix<double,2,3> bbox;

      //the bbox width
      double bbox_width;

      //margin so that no point gets lost
      double marg = 1e-3;

      if(axes_widths.maxCoeff() == axes_widths(0)){

        //the width of the bbox
        bbox_width = max_coeffs(0) - min_coeffs(0);
        //x
        bbox(0,0) = min_coeffs(0) - marg;
        bbox(1,0) = max_coeffs(0) + marg;
        //y
        bbox(0,1) = 0.5*(min_coeffs(1) + max_coeffs(1) - bbox_width) - marg;
        bbox(1,1) = 0.5*(min_coeffs(1) + max_coeffs(1) + bbox_width) + marg;
        //z
        bbox(0,2) = 0.5*(min_coeffs(2) + max_coeffs(2) - bbox_width) - marg;
        bbox(1,2) = 0.5*(min_coeffs(2) + max_coeffs(2) + bbox_width) + marg;



      }
      else if(axes_widths.maxCoeff() == axes_widths(1)){

        //the width of the bbox
        bbox_width = max_coeffs(1) - min_coeffs(1);
        //x
        bbox(0,0) = 0.5*(min_coeffs(0) + max_coeffs(0) - bbox_width) - marg;
        bbox(1,0) = 0.5*(min_coeffs(0) + max_coeffs(0) + bbox_width) + marg;
        //y
        bbox(0,1) = min_coeffs(1) - marg;
        bbox(1,1) = max_coeffs(1) + marg;
        //z
        bbox(0,2) = 0.5*(min_coeffs(2) + max_coeffs(2) - bbox_width) - marg;
        bbox(1,2) = 0.5*(min_coeffs(2) + max_coeffs(2) + bbox_width) + marg;
      }
      else if(axes_widths.maxCoeff() == axes_widths(2)){

        //the width of the bbox
        bbox_width = max_coeffs(2) - min_coeffs(2);
        //x
        bbox(0,0) = 0.5*(min_coeffs(0) + max_coeffs(0) - bbox_width) - marg;
        bbox(1,0) = 0.5*(min_coeffs(0) + max_coeffs(0) + bbox_width) + marg;
        //y
        bbox(0,1) = 0.5*(min_coeffs(1) + max_coeffs(1) - bbox_width) - marg;
        bbox(1,1) = 0.5*(min_coeffs(1) + max_coeffs(1) + bbox_width) + marg;
        //z
        bbox(0,2) = min_coeffs(2) - marg;
        bbox(1,2) = max_coeffs(2) + marg;
      }

      //std::cout << "bbox = " << bbox << std::endl;

      return bbox;

  }

  void compute_leaf_moments(){

    //element tree memory
    std::shared_ptr<ClusterTreeMemory> src_tree_mem = src_tree_.get_tree_memory();

    //here we run upwards, and look for the leafs in the element tree
    //the first level in the tree is labeled with -1. max_tree_lvl = 2 means that
    //there are the levels -1 , 0 , 1 and 2 so four levels. The last level is
    //at position max_tree_lvl_ +1 in the memory.
    for(int l = max_tree_lvl_+1; l > 1 ; --l){

    //#pragma omp parallel  TO DO: get this parallelized on Windows!
    {

    //#pragma omp for  TO DO: get this parallelized on Windows!
      for(std::vector<ClusterTreeNode>::iterator  src_it = src_tree_mem->lbegin(l); src_it < src_tree_mem->lend(l) ; src_it++)
      {

        //moments are computed for the leafs only (#sons = 0)
        if(src_it->sons_.size() == 0){

          //allocate the multipole moments
          src_it->moments_ = Eigen::MatrixXcd(num_coeffs_,3);

          //loop over all indices to sum up the contributions
          for (int j : src_it->indices_){


            //compute the solid Harmonics
            Eigen::VectorXcd R = Rlm_alt(L_,  src_.row(j).transpose() - src_it->center_);


            src_it->moments_.col(0) += R.conjugate()*src_vec_(j,0);
            src_it->moments_.col(1) += R.conjugate()*src_vec_(j,1);
            src_it->moments_.col(2) += R.conjugate()*src_vec_(j,2);



          }


          //remeber the position of this leaf in the memory
          //#pragma omp critical   TO DO: get this parallelized on Windows!
          {
            source_leafs_.push_back(src_it->pos_);


          }

        }
        }
      }
    }
  }

  void make_leafs_local_matrices(){


    //meas tree memory
    std::shared_ptr<ClusterTreeMemory> tar_tree_mem = tar_tree_.get_tree_memory();

    //--------------------------------------------------------------------------
    //Make leafs local matrices
    //--------------------------------------------------------------------------
    //std::cout << "  make leafs local matrices" << std::endl;
    for(int l = max_tree_lvl_+1; l >= 2 ; --l){



    //#pragma omp parallel for  TO DO: get this parallelized on Windows!
      for(std::vector<ClusterTreeNode>::iterator tar_it = tar_tree_mem->lbegin(l); tar_it < tar_tree_mem->lend(l) ; tar_it++){

        //locals are computed for the leafs only (#sons = 0)
        if(tar_it->sons_.size() == 0){

          //make space for local matrix
          tar_it->local_mat_.resize(dim_output_space_*num_tar_,3*num_coeffs_);

          //running index
          //int k = 0;

          //make tripletList to fill sparse moment matrix
          typedef Eigen::Triplet<std::complex<double>> T;
          std::vector<T> tripletList;
          tripletList.reserve(dim_output_space_*tar_it->indices_.size()*3*num_coeffs_);


          //iterate over the measurements of this cell
          for(int tar_indx: tar_it->indices_){


            //this is a matrix of dimension dim_output_space_*num_coeffs x 3
            //in the columns, the components of the output space are stored
            //in the rows, the fm components are found.
            Eigen::MatrixXcd LocalExp = monitor_.evaluateLocalExpansion(tar_.row(tar_indx),tar_it->center_,L_);


            for(int k = 0; k < num_coeffs_; ++ k){

              for (int i = 0; i < dim_output_space_; ++i){
                for( int j = 0; j < 3; ++j){


                  tripletList.push_back(T(tar_indx + i*num_tar_ , k + j*num_coeffs_ , LocalExp(k + i*num_coeffs_,j) ));

                }
              }
            }

          }



          tar_it->local_mat_.setFromTriplets(tripletList.begin(), tripletList.end());




        //remeber the position of this leaf in the memory
        //#pragma omp critical
        {

          local_leafs_.push_back(tar_it->pos_);
        }

        }

      }
    }

  }

  void upward_pass(){

      //element tree memory
      std::shared_ptr<ClusterTreeMemory> src_tree_mem = src_tree_.get_tree_memory();

      //we start from the second deepest level
      //we iterate up until level 1, this level is bisected twice and contains: 64
      //cells.
      for(int l = max_tree_lvl_; l > 1 ; --l){


  //#pragma omp parallel for  TO DO: get this parallelized on Windows!
        for(std::vector<ClusterTreeNode>::iterator  src_it = src_tree_mem->lbegin(l); src_it < src_tree_mem->lend(l) ; ++src_it){

          //std::cout << "Cell center = " << (*el_it).center_ << std::endl;

          //transformation is only performed to father cells
          if(src_it->sons_.size() > 0){


            //std::cout << "\tis father" << std::endl;

            //pointer to son
            ClusterTreeNode *son;

            //make space for moments in father
            src_it->moments_.resize(num_coeffs_,3);
            src_it->moments_.setZero();

            for(int i = 0; i < src_it->sons_.size(); ++i){


              son = src_tree_mem->get_element_ptr(src_it->sons_[i]);



              m2m_matrices_.m2m_mod(l, son->center_ - src_it->center_, son->moments_, &src_it->moments_);

              /*
              std::cout << std::endl;
              std::cout << "s ctr = \n" << son->center_.transpose()  << std::endl;
              std::cout << std::endl;
              std::cout << "f ctr = \n" << src_it->center_.transpose()  << std::endl;
              std::cout << std::endl;
              std::cout << "m_s real = \n" << son->moments_.real() << std::endl;
              std::cout << std::endl;
              std::cout << "m_f real = \n" << src_it->moments_.real() << std::endl;
              std::cout << std::endl;
              std::cout << "m_s imag = \n" << son->moments_.imag() << std::endl;
              std::cout << std::endl;
              std::cout << "m_f imag = \n" << src_it->moments_.imag() << std::endl;
              */

            }

          }
        }
      }
  }

  void downward_pass(){

    //we start from the second level. This level is at memory position 2
    for(int l = 1; l <= max_tree_lvl_+1 ; ++l){

    //#pragma omp parallel    TO DO: get this parallelized on Windows!
    {



      //source tree memory
      std::shared_ptr<ClusterTreeMemory> src_tree_mem = src_tree_.get_tree_memory();

      //target tree memory
      std::shared_ptr<ClusterTreeMemory> tar_tree_mem = tar_tree_.get_tree_memory();


      //#pragma omp for    TO DO: get this parallelized on Windows!
        for(std::vector<ClusterTreeNode>::iterator tar_it = tar_tree_mem->lbegin(l); tar_it < tar_tree_mem->lend(l) ; tar_it++){


          //pointer to source cell
          ClusterTreeNode *src_cell;

          //make space for locals
          tar_it->locals_.resize(num_coeffs_,3);
          //set locals to zero
          tar_it->locals_.setZero();





          //m2l transformations in the interaction region
          for(int i = 0; i < tar_it->interaction_region_.size(); ++i){


            src_cell = src_tree_mem->get_element_ptr(tar_it->interaction_region_[i]);


            //rotate
            Eigen::MatrixXcd tmp = m2l_rot_[tar_it->interaction_m2l_rot_index_[i]] * src_cell->moments_;

            //translate
            tmp = (m2l_tansl_[tar_it->interaction_m2l_index_[i]] * tmp).eval();

            //rotate back
            tar_it->locals_ += (m2l_rot_[tar_it->interaction_m2l_rot_index_[i]].adjoint() * tmp).eval();

            /*
            std::cout << "source moments:" << std::endl;
            print_moments(src_cell->moments_);
            std::cout << "target locals:" << std::endl;
            print_moments(tar_it->locals_);
            std::cout << "source center:" << std::endl;
            std::cout << src_cell->center_.transpose() << std::endl;
            std::cout << "target center:" << std::endl;
            std::cout << tar_it->center_.transpose() << std::endl;
            */
          }
          if(l > 2){

            //std::cout << "l2l" << std::endl;
            //l2l transformations for the far field

            //std::cout << "\tl2l" << std::endl;


            //pointer to father cell
            ClusterTreeNode *father = tar_tree_mem->get_element_ptr(tar_it->father_);

            //std::cout << "\t\tfather center = " << father->center_.transpose() << std::endl;
            //std::cout << "\t\tson center = " << meas_it->center_.transpose() << std::endl;



            //if(father->locals_.norm() > 1e-12){
              //l2l transformation
            l2l_matrices_.l2l_mod(l-1, father->center_ - tar_it->center_, father->locals_, &tar_it->locals_);

            /*
            std::cout << "father locals:" << std::endl;
            print_moments(father->locals_);
            std::cout << "son locals:" << std::endl;
            print_moments(tar_it->locals_);
            std::cout << "father center:" << std::endl;
            std::cout << father->center_.transpose() << std::endl;
            std::cout << "son center:" << std::endl;
            std::cout << tar_it->center_.transpose() << std::endl;
            */
            //}
          }

        }
      }
    }
  }

  void print_moments(const Eigen::MatrixXcd &moments){

    std::cout << std::endl;
    for (int k = 0; k < moments.rows(); ++k){

      std::cout << "[" << moments(k,0).real() << " + "  << moments(k,0).imag() << "*1j , " << moments(k,1).real() << " + "  << moments(k,1).imag()  << "*1j , " << moments(k,2).real() << " + "  << moments(k,2).imag()  << "*1j ]," << std::endl;
    }
    std::cout << std::endl;
  }

  Eigen::MatrixXd evaluate_far_field(){


      //make space for result vector
      Eigen::VectorXd result_flat(dim_output_space_*num_tar_);
      result_flat.setZero();



      #pragma omp parallel
      {

      //meas tree memory
      std::shared_ptr<ClusterTreeMemory> tar_tree_mem = tar_tree_.get_tree_memory();


      //measurement tree node
      ClusterTreeNode *tar_node;


      Eigen::VectorXd my_result(dim_output_space_*num_tar_);
      my_result.setZero();


      //make space for flattened locals
      Eigen::VectorXcd locals_flat(3*num_coeffs_);


      #pragma omp for
      for(int i = 0; i < local_leafs_.size(); ++i){

        //get the measurement node
        tar_node = tar_tree_mem->get_element_ptr(local_leafs_[i]);


        //we need to go back to a flat vector
        for(int j = 0; j < 3; ++j){
          locals_flat.segment(j*num_coeffs_,num_coeffs_) = tar_node->locals_.col(j);
        }


        //increment my result
        my_result += (tar_node->local_mat_*locals_flat).real();

      }

      #pragma omp critical
      {

        result_flat += my_result;
      }

    }


    //reshape
    Eigen::MatrixXd result(num_tar_,dim_output_space_);


    for(int i = 0; i < dim_output_space_;++i){

      result.col(i) = result_flat.segment(i*num_tar_,num_tar_);
    }


    //std::cout << "Far field norm = " << result.norm() << std::endl;
    return 1e-7*result;///4 ./BEMBEL_PI;


  }

  Eigen::MatrixXd evaluate_near_field(){


    //make space for result vector
    Eigen::MatrixXd result(num_tar_,dim_output_space_);
    result.setZero();

    #pragma omp parallel
    {

      //local and source indices
      int i_l,i_s;

      //pointers to tree memory
      std::shared_ptr<ClusterTreeMemory> src_tree_mem = src_tree_.get_tree_memory();
      std::shared_ptr<ClusterTreeMemory> tar_tree_mem = tar_tree_.get_tree_memory();

      //this is the result obtained from this cluster interaction
      Eigen::MatrixXd my_result(num_tar_,dim_output_space_);
      my_result.setZero();


      #pragma omp for
      for(int i = 0; i < near_field_interaction_list_.size() ; ++i){



        //indices of interaction
        i_l = near_field_interaction_list_[i][1];
        i_s = near_field_interaction_list_[i][0];

        //interacting cells
        ClusterTreeNode *src_node = src_tree_mem->get_element_ptr(i_s);
        ClusterTreeNode *tar_node = tar_tree_mem->get_element_ptr(i_l);

        for(int tar_indx : tar_node->indices_){


          for(int source_indx : src_node->indices_){



            my_result.row(tar_indx) +=  monitor_.evaluate_near_field_interaction(tar_.row(tar_indx),
                                                                                src_.row(source_indx),
                                                                                src_vec_.row(source_indx));


          }

          //std::cout << "tar_indx = " << tar_indx << std::endl;
          //std::cout << "   src_indices size = " << src_node->indices_.size() << std::endl;
          //std::cout << "   my_result.row = " << my_result.row(tar_indx) << std::endl;


        }


      }

      #pragma omp critical
      {

        result += my_result;
      }

    }
    //std::cout << "Far field norm = " << result.norm() << std::endl;
    return 1e-7*result;///4 ./BEMBEL_PI;


  }


  void make_ir_and_nf(){

    //element tree memory
    std::shared_ptr<ClusterTreeMemory> src_tree_mem = src_tree_.get_tree_memory();

    //meas tree memory
    std::shared_ptr<ClusterTreeMemory> tar_tree_mem = tar_tree_.get_tree_memory();

    //distance vector
    Eigen::Vector3d d_vec;
    //distance vector of fathers
    Eigen::Vector3d d_vec_f;

    //reset near field interaction count
    num_near_field_interactions_ = 0;

    //a distance variable
    double d;

    //father cells
    ClusterTreeNode src_father;
    ClusterTreeNode tar_father;

    for(int l = 2; l <= max_tree_lvl_+1 ; ++l){

      //std::cout << "level = " << l << std::endl;

      std::vector<ClusterTreeNode>::iterator src_it;
      std::vector<ClusterTreeNode>::iterator tar_it;

      for(src_it = src_tree_mem->lbegin(l); src_it != src_tree_mem->lend(l) ; src_it++)
      {


        src_father = src_tree_mem->get_element((*src_it).father_);

        for(tar_it = tar_tree_mem->lbegin(l); tar_it != tar_tree_mem->lend(l) ; tar_it++){


              tar_father = tar_tree_mem->get_element((*tar_it).father_);

              int c = compare_cells(l, (*src_it).center_ ,(*tar_it).center_,src_father.center_,tar_father.center_);

              //std::cout << "(*src_it).center_ = " << (*src_it).center_ << std::endl;
              //std::cout << "(*tar_it).center_ = " << (*tar_it).center_ << std::endl;
              //std::cout << "c = " << c << std::endl;

              if(c == 0){
                  //dense operation

                  //only leafs can interact as near field.
                  if((src_it->sons_.size() == 0) || (tar_it->sons_.size() == 0)){

                    (*tar_it).near_field_.push_back((*src_it).pos_);
                    (*src_it).near_field_.push_back((*tar_it).pos_);

                    //add the clusters to the near field interaction list
                    near_field_interaction_list_.push_back({(*src_it).pos_,(*tar_it).pos_});

                    //increment the number of near field interactions
                    num_near_field_interactions_ += (*tar_it).indices_.size() * (*src_it).indices_.size();
                  }
              }
              else if( c == 1 ){
                //compressed operation

                (*tar_it).interaction_region_.push_back((*src_it).pos_);
                (*src_it).interaction_region_.push_back((*tar_it).pos_);

                //Eigen::MatrixXcd tmp = make_m2l_matrix(meas_it->center_ - el_it->center_);

                //source - local
                //we are searching for the m2l matrix based on the difference vector
                Eigen::Vector3d d_tmp = src_it->center_ - tar_it->center_;

                //find rotation matrix
                int rot_indx = find_rot_matrix(d_tmp);

                //if not found, we append this rotation to the unique list
                if(rot_indx == -1){
                    m2l_rot_.push_back(make_m2l_rot(src_it->center_,tar_it->center_));
                    d_rot_vecs_.push_back(d_tmp);

                    rot_indx = m2l_rot_.size() - 1;
                }

                //find transfer matrix
                int m2l_indx = find_m2l_matrix(d_tmp);

                //if not found, we append this rotation to the unique list
                if(m2l_indx == -1){
                    m2l_tansl_.push_back(make_m2l_tansl(d_tmp.norm()));
                    d_vecs_.push_back(d_tmp);

                    m2l_indx = m2l_tansl_.size() - 1;
                }

                (*tar_it).interaction_m2l_index_.push_back(m2l_indx);
                (*tar_it).interaction_m2l_rot_index_.push_back(rot_indx);

                (*src_it).interaction_m2l_index_.push_back(m2l_indx);
                (*src_it).interaction_m2l_rot_index_.push_back(rot_indx);

              }
        }
      }
    }
  }

  int compare_cells(const int level, const Eigen::Vector3d &xs_1,const Eigen::Vector3d &xs_2,
                                      const Eigen::Vector3d &xf_1,const Eigen::Vector3d &xf_2){

    Eigen::Vector3d diff_s = xs_1 - xs_2;
    Eigen::Vector3d diff_f = xf_1 - xf_2;

    if((std::fabs(diff_s(0)) <= hx_ / std::pow(2,level) * 1.001 )
        && (std::fabs(diff_s(1)) <= hy_ / std::pow(2,level) * 1.001 )
        && (std::fabs(diff_s(2)) <= hz_ / std::pow(2,level) * 1.001 ) ){

          //std::cout << "found adjacent cells" << std::endl;
          //adjacent
          return 0;

        }
    else if(level == 2){

          //std::cout << "all other cells on level 2 are interacting" << std::endl;
          //interaction region, all are interacting in level 2
          return 1;
    }
    else if((std::fabs(diff_f(0)) <= hx_ / std::pow(2,level-1) * 1.001 )
        && (std::fabs(diff_f(1)) <= hy_ / std::pow(2,level-1) * 1.001 )
        && (std::fabs(diff_f(2)) <= hz_ / std::pow(2,level-1) * 1.001 )){

      //std::cout << "found interacting cells" << std::endl;
      //interaction region
      return 1;

    }
    else{

      //std::cout << "this is far field" << std::endl;
      //far field
      return 2;

    }

  }

  //Make a moment to local rotation matrix
  Eigen::SparseMatrix<std::complex<double>,Eigen::RowMajor> make_m2l_rot(const Eigen::Vector3d &x_s,const Eigen::Vector3d &x_l){

    Eigen::SparseMatrix<std::complex<double>,Eigen::RowMajor> ret_mat;
    ret_mat.resize(num_coeffs_,num_coeffs_);
    ret_mat.setZero();

    //source - local
    Eigen::Vector3d d = x_s - x_l;
    double r = d.norm();

    //algles
    double alpha = std::atan2(d(1),d(0));
    double beta = std::acos(d(2)/r);

    ret_mat =  make_rotation_matrix(L_,beta,alpha);

    return ret_mat;
  }

  //Make a moment to local translation matrix for translation along z
  Eigen::SparseMatrix<std::complex<double>,Eigen::RowMajor> make_m2l_tansl(const double dist){

    Eigen::SparseMatrix<std::complex<double>,Eigen::RowMajor> ret_mat;
    ret_mat.resize(num_coeffs_,num_coeffs_);
    ret_mat.setZero();

    int row;
    int col;

    std::complex<double> num;
    std::complex<double> den;

    //imaginary number as helper
    std::complex<double> I(0.0,1.0);

    //tripletList to construct sparse matrix
    typedef Eigen::Triplet<std::complex<double>> T;
    std::vector<T> tripletList;
    tripletList.reserve(num_coeffs_*num_coeffs_);

    for(int j = 0; j <= L_; ++j){
      for(int k = -j; k <= j; ++k){

        for(int n = 0; n <= L_; ++n){
          for(int m = -n; m <= n ; ++m){

            if(m == k){

              row = j*j + j + k;
              col = n*n + n + m;

              num = std::pow(-1.,-std::abs(m))*A_nm(n,m)*A_nm(j,k)*Ynm_alt(0.,0.,j+n,0);
              den = std::pow(-1.,n)* A_nm(j+n,0)*std::pow(dist,j+n+1);

              tripletList.push_back(T(row ,col, num/den ));

            }
          }
        }
      }
    }


    //std::cout << "make sparse matrix" << std::endl;
    ret_mat.setFromTriplets(tripletList.begin(), tripletList.end());
    //std::cout << ret_mat << std::endl;
    return ret_mat;
  }

  int find_m2l_matrix(const Eigen::Vector3d &d){

    int ret_val = -1;

    double dist = d.norm();


    for(int i = 0; i < d_vecs_.size(); ++i){


      if( std::abs(d_vecs_[i].norm() - dist) < 1e-10 ) {

        ret_val = i;
        break;
      }
    }
    return ret_val;
  }

  //source - local
  int find_rot_matrix(const Eigen::Vector3d &d){

    int ret_val = -1;

    //std::cout << "find_rot_matrix" << std::endl;

    for(int i = 0; i < d_rot_vecs_.size(); ++i){

      //std::cout << "d_rot = " << d_rot_vecs_[i].transpose() << std::endl;
      //std::cout << "d = " << d.transpose() << std::endl;

      double proj = d_rot_vecs_[i].transpose() * d;

      //std::cout << "proj = " << proj << std::endl;

      if( std::abs(proj/d.norm()/d_rot_vecs_[i].norm()  - 1.) < 1e-10){

        ret_val = i;
        break;
      }
    }
    return ret_val;
  }

  void compute_box_diam(const Eigen::Matrix<double,2,3> &bbox){

    diag_ = (bbox.row(1)-bbox.row(0)).transpose();
    diam_ = diag_.norm();

    hx_ = bbox(1,0) - bbox(0,0);
    hy_ = bbox(1,1) - bbox(0,1);
    hz_ = bbox(1,2) - bbox(0,2);

  }

 };


#endif
