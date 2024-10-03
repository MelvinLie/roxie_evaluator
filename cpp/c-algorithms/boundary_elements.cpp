#include <stdlib.h>
#include <malloc.h>
#include <iostream>

void evaluate_boundary_element(double *pos, const int* c, const int c_idx, const int num_nodes, const double *nodes, const double *basis, const int num_pos)
{


  /**
  * Evaluate a boundary element.

  * @param pos The evaluated position. Make sure that is array is
               allocated with zeros!
  * @param c The enode connectivity, as integer array.
  * @param c_idx The current index in the cell array.
  * @param num_nodes The number of nodes for this finite element.
  * @param nodes The mesh nodal coordinates.
  * @param basis The basis functions of this boundary element. Sorted like this:
                    basis = [N1(p1), N2(p1), ..., N1(p2), ...]
  * @param num_pos The number of evaluation position.
  * @return Nothing.
  */

  // the loop indices
  int i, j;

  // set zero
  for (j = 0; j < num_pos; ++j){
    pos[3*j    ] = 0.0;
    pos[3*j + 1] = 0.0;
    pos[3*j + 2] = 0.0;
  }

  // loop over all nodes
  for (i = 0; i < num_nodes; ++i){

    // loop over all evaluation positions
    for (j = 0; j < num_pos; ++j){

      pos[3*j    ] += basis[i + j*num_nodes]*nodes[3*c[c_idx+i]    ];
      pos[3*j + 1] += basis[i + j*num_nodes]*nodes[3*c[c_idx+i] + 1];
      pos[3*j + 2] += basis[i + j*num_nodes]*nodes[3*c[c_idx+i] + 2];

    }
  }
  return;
}

void evaluate_boundary_element_derivatives(double *der, const int* c, const int c_idx, const int num_nodes, const double *nodes, const double *basis_der, const int num_pos){

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
                    basis_der = [dN1du(p1), dN1dv(p1), 0.0, dN2du(p1), ..., dN2du(p2),]
                    gmsh gives always three components, also for surface elements.
    * @param num_pos The number of evaluation position.
    * @return Nothing.
  */

  // the loop indices
  int i, j;
  

  // loop over all nodes
  for (i = 0; i < num_nodes; ++i){

    // loop over all evaluation positions
    for (j = 0; j < num_pos; ++j){

      der[2*(3*j    )    ] += basis_der[3*(i + j*num_nodes)    ]*nodes[3*c[c_idx+i]    ];
      der[2*(3*j + 1)    ] += basis_der[3*(i + j*num_nodes)    ]*nodes[3*c[c_idx+i] + 1];
      der[2*(3*j + 2)    ] += basis_der[3*(i + j*num_nodes)    ]*nodes[3*c[c_idx+i] + 2];

      der[2*(3*j    ) + 1] += basis_der[3*(i + j*num_nodes) + 1]*nodes[3*c[c_idx+i]    ];
      der[2*(3*j + 1) + 1] += basis_der[3*(i + j*num_nodes) + 1]*nodes[3*c[c_idx+i] + 1];
      der[2*(3*j + 2) + 1] += basis_der[3*(i + j*num_nodes) + 1]*nodes[3*c[c_idx+i] + 2];
    }
  }
  return;
}

void evaluate_surface_curls(double *curl, const int* c, const int c_idx, const int num_nodes, const double *nodes, const double *basis_der, const int num_pos){

  /**
    * Evaluate the surface curls.

    * @param curl The evaluated surface curl. They are sorted like this:
                    curl = [curlN1x(p1), curlN1y(p1), curlN1z(p1), curlN2x(p1), ...., curlN1x(p2)]
    * @param c The enode connectivity, as integer array.
    * @param c_idx The current index in the cell array.
    * @param num_nodes The number of nodes for this finite element.
    * @param nodes The mesh nodal coordinates.
                    nodes = [x1, y1, z1, x2, y2, z2, ...]
    * @param basis_der The basis function derivatives of this boundary element. Sorted like this:
                    basis_der = [dN1du(p1), dN1dv(p1), 0.0 dN2du(p1), ..., dN2du(p2),]
                    gmsh gives always three components, also for surface elements.
    * @param num_pos The number of evaluation position.
    * @return Nothing
  */
  
  // the loop indices
  int i, j;

  // the derivatives evaluated on this element
  double *der;

  // allocate and zero the derivative pointer
  der = (double*)calloc(6*num_pos, sizeof(double));

  // compute the derivatives
  evaluate_boundary_element_derivatives(der, c, c_idx, num_nodes, nodes, basis_der, num_pos);

  // set zero
  for (i = 0; i < 3*num_pos*num_nodes; ++i){
    curl[i] = 0.0;
  }

  // loop over all nodes
  for (i = 0; i < num_nodes; ++i){

    // loop over all evaluation positions
    for (j = 0; j < num_pos; ++j){

      curl[3*(j*num_nodes + i) + 0] = der[6*j + 0]*basis_der[3*(num_nodes*j + i) + 1] - der[6*j + 1]*basis_der[3*(num_nodes*j + i)];
      curl[3*(j*num_nodes + i) + 1] = der[6*j + 2]*basis_der[3*(num_nodes*j + i) + 1] - der[6*j + 3]*basis_der[3*(num_nodes*j + i)];
      curl[3*(j*num_nodes + i) + 2] = der[6*j + 4]*basis_der[3*(num_nodes*j + i) + 1] - der[6*j + 5]*basis_der[3*(num_nodes*j + i)];

    }
  }

  return;
}