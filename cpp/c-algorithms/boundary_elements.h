#ifndef BOUNDARY_ELEMENTS_H_
#define BOUNDARY_ELEMENTS_H_



/**
  * Evaluate a boundary element.

  * @param pos The evaluated position.
  * @param c The enode connectivity, as integer array.
  * @param c_idx The current index in the cell array.
  * @param num_nodes The number of nodes for this finite element.
  * @param nodes The mesh nodal coordinates.
  * @param basis The basis functions of this boundary element.
  * @param num_pos The number of evaluation position.
  * @return nodes
*/
void evaluate_boundary_element(double *pos, const int* c, const int c_idx, const int num_nodes, const double *nodes, const double *basis, const int num_pos);

/**
  * Evaluate the derivatives of a boundary element parameterization.

  * @param der The evaluated derivatives. Sorted like this:
                    der = [dxdu(p1), dxdv(p1), dydu(p1), dydv(p1), dzdu(p1), dzdv(p1), dxdu(p2), ...]
  * @param c The enode connectivity, as integer array.
  * @param c_idx The current index in the cell array.
  * @param num_nodes The number of nodes for this finite element.
  * @param nodes The mesh nodal coordinates. Sorted as follows:
                    nodes = [x1, y1, z1, x2, y2, z2, ...]
  * @param basis_der The basis function derivatives of this boundary element. Sorted like this:
                    basis_der = [dN1du(p1), dN1dv(p1), dN1du(p1), ..., dN2du(p2),]
  * @param num_pos The number of evaluation position.
  * @return Nothing.
*/
void evaluate_boundary_element_derivatives(double *pos, const int* c, const int c_idx, const int num_nodes, const double *nodes, const double *basis_der, const int num_pos);

/**
  * Evaluate the surface curls.

  * @param curl The evaluated surface curl.
  * @param c The enode connectivity, as integer array.
  * @param c_idx The current index in the cell array.
  * @param num_nodes The number of nodes for this finite element.
  * @param nodes The mesh nodal coordinates.
  * @param basis The basis function derivatives of this boundary element.
  * @param num_pos The number of evaluation position.
  * @return nodes
*/
void evaluate_surface_curls(double *curl, const int* c, const int c_idx, const int num_nodes, const double *nodes, const double *basis_der, const int num_pos);

#endif
