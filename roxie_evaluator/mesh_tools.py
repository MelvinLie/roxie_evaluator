import numpy as np
import gmsh
import pyvista as pv

from . import integration_tools as int_tools
from . import hexahedral_elements as hex

def compute_jacobian_determinants(nodes, cells, quad_order=5, mesh_format='hmo', loc=np.zeros((0, 3))):
    '''Compute the determinants of the Jacobians for
    a given hexahedral mesh.

    :param nodes:
        The mesh nodal coordinates.

    :param cells:
        The connectivity of the mesh.

    :param quad_order:
        The quadrature order. The number of local evaluation points
        is selected accordingly.

    :param mesh_format:
        The mesh format string specifies where the mesh is coming from.
        The string "hmo" must be used if an hmo file was read.

    :param loc:
        An array with local coordinates to evaluate. If empty (default),
        the quadrature order will determine the points.

    :return:
        The determinants of the Jacobians evaluated at the
        quadrature points, as well as the global coordinates of the
        quadrature points.
    '''

    # make a local copy of the cell array
    cells_c = cells.copy()

    # hmo node numbers start with 1
    if mesh_format == 'hmo':
        cells_c -= 1
    else:
        print('Unknown mesh format {}! Only hmo is implemented yet!'.format(mesh_format))
        return
    
    # get the number of boundary elements
    num_el = cells_c.shape[0]

    if loc.shape[0] == 0:
        # get the quadrature points
        q, w = int_tools.get_quadrature_rule(quad_order)
    else:
        q = loc

    # the number of points per element
    num_pts = q.shape[0]

    # the positions
    points = np.zeros((num_el*num_pts, 3))

    # the vector of jacobian determinants
    det_J = np.zeros((num_el*num_pts, ))

    # evaluate the shape functions at these points (not needed)
    phi = hex.eval_shape_hexahedron(q, 2)



    # evaluate the derivatives at these points
    d_phi = hex.eval_gradient_hexahedron(q, 2)
        
    # the gmsh node definition is different to the one used in hypermesh
    if mesh_format == 'hmo':
        indx = np.array([0, 8, 1, 11, 2, 13, 3, 9, 10, 12, 14, 15, 4, 16, 5, 18, 6, 19, 7, 17], dtype=np.int64)


        # sort
        phi = phi[:, indx]
        d_phi = d_phi[:, indx, :]

    else:
        print('Unknown mesh format {}! Only hmo is implemented yet!'.format(mesh_format))
        return
    

    # loop over all elements
    for i, e in enumerate(cells_c):
            
        # evaluate this hexahedron for the global position
        points[i*num_pts:(i+1)*num_pts, :] = hex.evaluate_hexahedron(e, nodes, phi)

        # compute the Jabcobi matrix
        J = hex.compute_J(e, nodes, d_phi)

        # invert it
        # inv_J = hex.compute_J_inv(J)

        # compute its determinant
        det_J[i*num_pts:(i+1)*num_pts] = hex.compute_J_det(J)


    return det_J, points


def make_box_gmsh(center, W, H, L, chamfer=0.0, lc=1.0):
    '''Make a box with gmsh.

    :param center:
        The center of the box.

    :param W:
        The box width.

    :param H:
        The box height.
    
    :param L:
        The box length.

    :param chamfer:
        Set the chamfer distance. If 0.0 (default). The edges will
        not be chamfered.

    :param lc:
        A gmsh mesh parameter applied to the corner points. Default = 1.0.
        
    :return:
        The gmsh model.
    '''

    # initialize gmsh if not done already
    if not gmsh.isInitialized():
        gmsh.initialize()

    # make the 8 corner points
    p1 = gmsh.model.occ.add_point(-0.5*W+center[0], -0.5*H+center[1], -0.5*L+center[2], lc)
    p2 = gmsh.model.occ.add_point(0.5*W+center[0], -0.5*H+center[1], -0.5*L+center[2], lc)
    p3 = gmsh.model.occ.add_point(0.5*W+center[0], 0.5*H+center[1], -0.5*L+center[2], lc)
    p4 = gmsh.model.occ.add_point(-0.5*W+center[0], 0.5*H+center[1], -0.5*L+center[2], lc)
    p5 = gmsh.model.occ.add_point(-0.5*W+center[0], -0.5*H+center[1], 0.5*L+center[2], lc)
    p6 = gmsh.model.occ.add_point(0.5*W+center[0], -0.5*H+center[1], 0.5*L+center[2], lc)
    p7 = gmsh.model.occ.add_point(0.5*W+center[0], 0.5*H+center[1], 0.5*L+center[2], lc)
    p8 = gmsh.model.occ.add_point(-0.5*W+center[0], 0.5*H+center[1], 0.5*L+center[2], lc)

    # make the 12 edges
    l1 = gmsh.model.occ.add_line(p1, p2)
    l2 = gmsh.model.occ.add_line(p2, p3)
    l3 = gmsh.model.occ.add_line(p3, p4)
    l4 = gmsh.model.occ.add_line(p4, p1)
    l5 = gmsh.model.occ.add_line(p5, p6)
    l6 = gmsh.model.occ.add_line(p6, p7)
    l7 = gmsh.model.occ.add_line(p7, p8)
    l8 = gmsh.model.occ.add_line(p8, p5)
    l9 = gmsh.model.occ.add_line(p1, p5)
    l10 = gmsh.model.occ.add_line(p2, p6)
    l11 = gmsh.model.occ.add_line(p3, p7)
    l12 = gmsh.model.occ.add_line(p4, p8)

    # make wires for the 6 sides
    w1 = gmsh.model.occ.add_wire([-l4, -l3, -l2, -l1])
    w2 = gmsh.model.occ.add_wire([l5, l6, l7, l8])
    w3 = gmsh.model.occ.add_wire([l9, l8, l12, l4])
    w4 = gmsh.model.occ.add_wire([l1, l10, l5, l9]) 
    w5 = gmsh.model.occ.add_wire([l2, l11, l6, l10])
    w6 = gmsh.model.occ.add_wire([l3, l12, l7, l11])

    # make the 6 sides
    s1 = gmsh.model.occ.add_plane_surface([w1])
    s2 = gmsh.model.occ.add_plane_surface([w2])
    s3 = gmsh.model.occ.add_plane_surface([w3])
    s4 = gmsh.model.occ.add_plane_surface([w4])
    s5 = gmsh.model.occ.add_plane_surface([w5])
    s6 = gmsh.model.occ.add_plane_surface([w6])

    # make surface loop
    sl = gmsh.model.occ.add_surface_loop([s1, s2, s3, s4, s5, s6])

    # make volume
    vol = gmsh.model.occ.add_volume([sl])

    gmsh.model.occ.synchronize()

    if chamfer > 0.0:
        gmsh.model.occ.chamfer([vol],
                            [l11, l7, l12, l3, l9, l5, l10, l1, l8, l6, l2, l4],
                            [s6, s6, s6, s6, s5, s5, s5, s5, s3, s2, s1, s1],
                            [chamfer, chamfer, chamfer, chamfer, chamfer, chamfer, chamfer, chamfer, chamfer, chamfer, chamfer, chamfer]) 

    gmsh.model.occ.synchronize()


    return gmsh.model


def gmsh_to_pyvista_2D(nodes, cells, gmsh_types):
    '''Make an UnstructuredGrid for plotting with pyvista based on a
    gmsh surface mesh.
    
    :param nodes:
        The nodal coordinates.
    
    :param cells:
        A list of mesh connectivity matrices.
        
    :param gmsh_type:
        A list of gmsh cell types.

    :return:
        None    
    '''

    # make a list of vtk cells
    vtk_cell_list = np.array([], dtype=np.int32)

    # make also a list of vtk cell types
    vtk_types_list = np.array([], dtype=np.int32)

    # loop over the cell types
    for i, ct in enumerate(gmsh_types):
        

        # get the number of nodes per cell
        # also get the corresponding vtk type
        # also make the stancil to convert the node ordering

        if ct == 2:
            # 3 noded triangle
            num_cell_nodes = 3
            vtk_type = 5
            stencil = [0, 1, 2]
        elif ct == 3:
            # 4 noded quad
            num_cell_nodes = 4
            vtk_type = 8
            stencil = [0, 1, 3, 2]
        elif ct == 9:
            # 6 noded triangle
            num_cell_nodes = 6
            vtk_type = 22
            stencil = [0, 1, 2, 3, 4, 5]
        elif ct == 10:
            # 9 noded quad
            num_cell_nodes = 9
            vtk_type = 28
            stencil = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        elif ct == 16:
            # 8 noded quad
            num_cell_nodes = 8
            vtk_type = 23
            stencil = [0, 1, 2, 3, 4, 5, 6, 7]
        else:
            print('Conversion for element of gmsh type {} unknown!'.format(gmsh_type))

        tmp_cells = cells[i].copy()
        num_cells = np.int32(len(tmp_cells)/num_cell_nodes)
        tmp_cells.shape = (num_cells, num_cell_nodes)

        vtk_cells = np.zeros((num_cells, num_cell_nodes+1), dtype=np.int32)
        vtk_cells[:, 1:] = tmp_cells[:, stencil] - 1
        vtk_cells[:, 0] = num_cell_nodes

        vtk_cell_list = np.append(vtk_cell_list, vtk_cells.flatten())
        vtk_types_list = np.append(vtk_types_list, np.array([vtk_type for i in range(num_cells)], dtype=np.int32))

    return pv.UnstructuredGrid(vtk_cell_list, vtk_types_list, nodes)
