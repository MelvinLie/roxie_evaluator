import numpy as np

def read_hmo_file(filename):
    '''Read an hmo file and return the mesh as nodes and connectivity.

    :param filename:
        The filename to read.

    :return:
        The component names. The node numbers.
        The nodal coordinates. The connectivity.
        And the element codes.
    '''

    # open the file
    hmo_file = open(filename, 'r')

    # read first line (version info etc)
    line = hmo_file.readline()

    # arm the readings
    arm_comp_data = False
    arm_node_data = False
    arm_elem_data = False

    # number of components
    num_comp = 0

    # a counter variable
    counter = 0

    # component names
    comp_names = []

    # containers for node numbers
    node_numbers = np.zeros((0,),dtype=np.int32)

    # containers for node info
    nodes = np.zeros((0,3))

    # containers for element info
    elements = np.zeros((0,))

    # element codes
    element_codes = []

    while True:

        # Get next line from file
        line = hmo_file.readline()


        # if line is empty
        # end of file is reached
        if not line:
            break

        if line == 'BEG_COMP_DATA\n':

            arm_comp_data = True


        elif(arm_comp_data):
            # parse this line
            parsed_line = [s for s in line.split(' ') if s]

            if(len(parsed_line) == 1) and line != 'END_COMP_DATA\n':

                # get the number of components
                num_comp = np.int32(parsed_line[0])

            elif line != 'END_COMP_DATA\n':

                if len(parsed_line) > 2:
                    comp_names.append(parsed_line[2][:-2])
                else:
                    comp_names.append(parsed_line[1][:-2])

            else:

                arm_comp_data = False

        if line == 'BEG_NODL_DATA\n':

            arm_node_data = True

        elif(arm_node_data):

            # parse this line
            parsed_line = [s for s in line.split(' ') if s]

            if(len(parsed_line) == 1) and line != 'END_NODL_DATA\n':

                # read the total number of nodes
                num_nodes = np.int32(parsed_line[0])

                # reallocate space for nodes
                nodes = np.zeros((num_nodes,3))
                node_numbers = np.zeros((num_nodes,),dtype = np.int32)

                # reset the counter
                counter = 0

            elif line != 'END_NODL_DATA\n':

                node_numbers[counter] = np.int32(parsed_line[0])
                nodes[counter, 0] = float(parsed_line[1])
                nodes[counter, 1] = float(parsed_line[2])
                nodes[counter, 2] = float(parsed_line[3])

                counter += 1

            else:

                arm_node_data = False

                # reset the counter
                counter = 0

        if line == 'BEG_ELEM_DATA\n':

            arm_elem_data = True

            # first line after this
            is_first_line = True


        elif(arm_elem_data) and line != 'END_ELEM_DATA\n':

            # parse this line
            parsed_line = [s for s in line.split(' ') if s]


            if is_first_line:

                # number of elements
                num_elem = np.int32(parsed_line[0])

                # reset the counter
                counter = 0

                is_first_line = False

            else:

                if counter == 0:
                    elements = np.zeros((num_elem,20),dtype=np.int32) - 1

                # number of nodes per element
                num_nodes_per_element = get_number_of_nodes(np.int32(parsed_line[2]))

                # append the element code
                element_codes.append(np.int32(parsed_line[2]))

                # reallocate space for elements
                for i in range(num_nodes_per_element):
                    elements[counter,i] = np.int32(parsed_line[i+3])

                counter += 1

        else:

            arm_elem_data = False

    hmo_file.close()

    return comp_names, node_numbers, nodes, elements, element_codes


def get_number_of_nodes(code):
    '''For a given EDYSON element code, return the number
    of nodes.

    :param code:
        The EDYSON element code.

    :return:
        The number of nodes.
    '''

    if code == 60:
        return 2
    elif code == 63:
        return 3
    elif code == 103:
        return 3
    elif code == 106:
        return 6
    elif code == 104:
        return 4
    elif code == 108:
        return 8
    elif code == 204:
        return 4
    elif code == 210:
        return 10
    elif code == 206:
        return 6
    elif code == 215:
        return 15
    elif code == 208:
        return 8
    elif code == 220:
        return 20
    
def get_vtk_cell_code(num_nodes, dim=3):
    '''Get the vtk cell code given the number of nodes.

    :param num_nodes:
        The number of nodes of the element.

    :param dim:
        The dimension of the element. Default 3.
        
    :return:
        The vtk element code.
    '''

    # https://docs.pyvista.org/version/stable/api/utilities/_autosummary/pyvista.celltype#pyvista.CellType

    if dim == 3:
        if num_nodes == 4:
            # 4 noded tetrahedral element
            return 10
        elif num_nodes == 8:
            # 8 noded brick element
            return 11
        elif num_nodes == 10:
            # 10 noded tetrahedral element
            return 24
        elif num_nodes == 20:
            # 20 noded brick element
            return 25
        elif num_nodes == 15:
            # 15 15 noded wedge element
            return 26
        else:
            print('Unknown number of nodes {}, for dimension {}!'.format(num_nodes, dim))
        
    elif dim == 2:
        if num_nodes == 6:
            # 6 noded triangle
            return 22
        elif num_nodes == 8:
            # 8 noded quad
            return 23
        else:
            print('Unknown number of nodes {}, for dimension {}!'.format(num_nodes, dim))

    else:
        print('Unknown dimendion of {}'.format(dim))