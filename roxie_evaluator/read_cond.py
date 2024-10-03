import numpy as np


def read_cond_file(filename):
    '''Read a cond file and get the conductor 
    geometry information.

    :param filename:
        The filename.

    :return:
        The mesh as nodes and connectivity.
    '''

    with open(filename, 'r') as fp:
        for count, line in enumerate(fp):
            pass
    print('Total Lines', count + 1)

    n_rows = count + 1

    # the number of bricks
    num_bricks = np.int32((n_rows-2)/15)

    print('number of bricks = {}'.format(num_bricks))

    # number of nodes
    num_nodes = 8*num_bricks

    # open the file
    brick_file = open(filename, 'r')

    # arm the readings
    arm_brick = False

    # flag to continue reading
    file_end = False

    # the list of points
    nodes = np.zeros((num_nodes, 3))

    # the currents
    currents = np.zeros((num_bricks, ))

    # the cells
    cells = np.zeros((num_bricks, 8), dtype=np.int32)

    # this is the stancil
    stancil = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int32)

    # the total number of nodes
    num_nodes = 0

    # a brick counter
    brick_count = 0

    while file_end == False:

        # Get next line from file
        line = brick_file.readline()

        # parse the line
        parsed_line = [s for s in line.split(' ') if s]

        if parsed_line[0] == 'DEFINE':
            if parsed_line[1][:-1] == 'BR8':

                # we found an 8 noded brick!

                # read 3 lines
                for i in range(3):
                    line = brick_file.readline()
                
                # read 8 lines
                for i in range(8):
                    line = brick_file.readline()
                    parsed_line = [s for s in line.split(' ') if s]


                    nodes[8*brick_count + i] = np.array([float(parsed_line[0]),
                                                        float(parsed_line[1]),
                                                        float(parsed_line[2])])


                cells[brick_count, :] = stancil + num_nodes

                num_nodes += 8

                # read the current
                line = brick_file.readline()
                parsed_line = [s for s in line.split('/') if s]
                currents[brick_count] = float(parsed_line[0])

                # read 2 lines
                for i in range(2):
                    line = brick_file.readline()

                brick_count += 1

        elif parsed_line[0][:-1] == 'QUIT':
            file_end = True


    return nodes, cells, currents