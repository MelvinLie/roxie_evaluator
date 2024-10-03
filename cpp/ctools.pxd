ctypedef fused DenseTypeShort:
	MatrixXd
	MatrixXi
	VectorXd
	Vector3d
	
cdef cppclass PlainObjectBase:
	pass

cdef cppclass MatrixXd(PlainObjectBase):
	pass

cdef cppclass MatrixXi(PlainObjectBase):
	pass

cdef cppclass VectorXd(PlainObjectBase):
	pass

cdef cppclass Vector3d(PlainObjectBase):
	pass

cdef extern from "boundary_elements.h":
	void evaluate_boundary_element(double *pos, const int* c, const int c_idx, const int num_nodes, const double *nodes, const double *basis, const int num_pos)
	void evaluate_boundary_element_derivatives(double *pos, const int* c, const int c_idx, const int num_nodes, const double *nodes, const double *basis_der, const int num_pos)
	void evaluate_surface_curls(double *curl, const int* c, const int c_idx, const int num_nodes, const double *nodes, const double *basis_der, const int num_pos)

cdef extern from "ctools.h":

	void test_eigen(double test)

cdef extern from "convertors.h":

	MatrixXd build_MatrixXd(const double *data, const int rows, const int cols)
	MatrixXi build_MatrixXi(const int *data, const int rows, const int cols)

	VectorXd build_VectorXd(const double *data, const int rows)

	void to_c_array(const MatrixXd &mat, double* data_ptr)

cdef extern from "analytical.h":

	Vector3d mvp_integral_current_segment(const Vector3d &r1, const Vector3d &r2, const double current)
	Vector3d mfd_integral_current_segment(const Vector3d &r1, const Vector3d &r2)
	void compute_integrals_current_segment(const Vector3d &r1, const Vector3d &r2, MatrixXd *A_mat, MatrixXd *B_mat, const double current, const int obs_index)

cdef extern from "mesh_tools.h":

	double compute_diam_src(const MatrixXd &src)
	double compute_diam_segments(const MatrixXd &segments)
	Vector3d eval_polygon(const double u, const double v, const Vector3d &p1, const Vector3d &p2, const Vector3d &p3, const Vector3d &p4)
	MatrixXd get_1D_gauss_integration_points(const int num_points)

cdef extern from "inductance_calculation.h":

	double compute_self_inductance_cylinder(const double length, const double radius)
	double compute_self_inductance_adjacent_straight_elements(const Vector3d &d1, const Vector3d &d2, const double radius)
	double inv_dist(const double xi1, const double xi2, const Vector3d &p11, const Vector3d &p12, const Vector3d &p21, const Vector3d &p22)
	double compute_interaction(const MatrixXd &Q, const Vector3d &p11, const Vector3d &p12, const Vector3d &p21, const Vector3d &p22)
	double compute_self_inductance(const MatrixXd &segments, const double radius, const int num_points, int is_open)

cdef extern from "evaluators.h":

	void compute_B_line_segs(double *ret_data, const double *src_ptr, const double *tar_ptr, const double current, const int num_src, const int num_tar)
	void compute_A_and_B(double* ret_data_A, double* ret_data_B, const double *p, const int *c, const int *N_n, const int *N_b, const double *I_strand, const double *tar_ptr, const int num_points, const  int num_bricks, const int num_tar, const double near_field_distance)
	void compute_B(double* ret_data_B, const double *p, const int *c, const int *N_n, const int *N_b, const double *I_strand, const double *tar_ptr, const int num_points, const int num_bricks, const int num_tar, const double near_field_distance)

	void compute_A_mlfmm(double *A, const int num_tar, const int num_src, const double *src_pts, const double *src_vec, const double *tar_pts, const int num_coeffs, const int max_tree_lvl, const double *b_box_c, const int potential_spec)
	void compute_B_mlfmm(double *B, const int num_tar, const int num_src, const double *src_pts, const double *src_vec, const double *tar_pts, const int num_coeffs, const int max_tree_lvl, const double *b_box_c, const int potential_spec)

	void compute_B_roxie_mlfmm(double *B, const int num_tar, const int num_src_coil, const double *src_pts_coil, const double *src_vec_coil, const int num_src_iron, const double *src_pts_iron, const double *src_vec_A, const double *src_vec_dA, const double *tar_pts, const int L, const int max_tree_lvl, const double *b_box_c)

	void compute_solid_harmonics(double *ret_real, double *ret_imag, const double *p, const int *c, const int *N_n, const int *N_b, const double *I_strand, const double *tar_ptr, const int num_points, const int num_bricks, const int num_tar, const int L, const int r_ref, const int num_quad_points, const double near_field_distance)
	
	double compute_L(const double *segs, const int num_segs, const double radius, const int num_points, const int is_open)

	void compute_A_iron(double *ret_A, const int num_pnt, const int num_src, const double *pnt, const double *q, const double *n, const double *w, const double *A,  const double *dA)
	void compute_B_iron(double *ret_B, const int num_pnt, const int num_src, const double *pnt, const double *q, const double *n, const double *w, const double *A,  const double *dA)

	void compute_B_eddy_ring(double *ret_B, const int num_pnt, const int num_src, const double *pnt, const double *q, const double *s, const double *w)
	void compute_B_eddy_ring_mat(double *ret_mat, const int num_pnt, const double *pnt, const double *n, const int num_nodes, const double *nodes, const int num_cells, const int *cells, const int num_cell_nodes, const double *basis, const double *basis_der, const int num_quad, const double *q, const double *w)