import numpy as np
import pandas as pd


def write_brick_vtk_file(filename, points, cells, cell_types, node_data=np.array([]), cell_data=np.array([])):
    '''Write a vtk file with mesh and nodal and cell data.

    :param filename:
        The output filename.

    :param points:
        The nodal coordinates.

    :param cells:
        The connectivity of the finite elements.

    :param cell_types:
        The cell types according to EDYSON.

    :param node_data:
        The nodal data.

    :param cell_data:
        The cell data.

    :return:
        Nothing.
    '''

    # number of points
    num_points = points.shape[0]

    # number of bricks
    num_bricks = cells.shape[0]

    # count how many numbers there are in the cell data
    cnt = 0

    for i in range(len(cell_types)):
        if cell_types[i] == 220:
            cnt += 21
        elif cell_types[i] == 215:
            cnt += 16

    # open the file
    with open(filename, 'w') as f:

        f.write("# vtk DataFile Version 2.0\n")
        f.write("CCT coil\n")
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")
        f.write("POINTS {} float\n".format(num_points))


        for i in range(num_points):

            f.write("{0:.12f} {1:.12f} {2:.12f}\n".format(points[i, 0], points[i, 1], points[i, 2]))

        f.write("CELLS {} {}\n".format(num_bricks, cnt))

        for i in range(num_bricks):

            # get the element type
            if cell_types[i] == 220:
                # 20 noded brick
                num_nodes = 20

                # stencil = [0, 8, 1, 11, 2, 13, 3, 9, 10, 12, 14, 15, 4, 16, 5, 18, 6, 19, 7, 17]
                # stencil = [0, 2, 4, 6, 12, 14, 16, 18, 1, 3, 5, 7, 13, 15, 17, 19, 8, 9, 10, 11]
                stencil = [0, 2, 4, 6, 12, 14, 16, 18, 1, 3, 5, 7, 13, 15, 17, 19, 8, 9, 10, 11]


            elif cell_types[i] == 215:
                # 15 noded wedge
                num_nodes = 15
                    
                stencil = [0, 2, 4, 9, 11, 13, 1, 3, 5, 10, 12, 14, 6, 7, 8]
                
            f.write("{}".format(num_nodes))

            for j in range(num_nodes):
                f.write(" {}".format(cells[i, stencil[j]] - 1))

            f.write("\n")

        f.write("CELL_TYPES {}\n".format(num_bricks))

        for i in range(num_bricks):
            # get the element type
            if cell_types[i] == 220:
                # 20 noded brick
                f.write("{}\n".format(25))

            elif cell_types[i] == 215:
                # 15 noded wedge
                f.write("{}\n".format(26))

        if len(cell_data) == num_bricks:

            f.write("CELL_DATA {}\n".format(num_bricks))
            f.write("SCALARS scalars float 1\n")
            f.write("LOOKUP_TABLE default\n")
            for cd in cell_data:
                f.write("{}\n".format(cd))

    return