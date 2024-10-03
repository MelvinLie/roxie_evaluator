import numpy as np

from .read_hmo import get_number_of_nodes

def read_ele_file(filename, num_cells):
    '''Read an ele file and return connectivity of the boundary domain.

    :param filename:
        The filename to read.

    :param num_cells:
        The number of cells in the hmo file.

    :return:
        The connectivity.
        The component names. 
        And the element codes.
    '''

    # open the file
    ele_file = open(filename, 'r')

    # read first line (version info etc)
    line = ele_file.readline()

    # second line is also irrelevant
    line = ele_file.readline()

    # then there is the information about the interior mesh
    for i in range(num_cells):
        line = ele_file.readline()

    # the connectivity list
    c = []
    comp_names = []
    element_codes = []

    # now read the actual data
    while (True):
        
        # read this line
        line = ele_file.readline()

        # parse it
        parsed_line = [s for s in line.split(' ') if s]

        # check end
        if(np.int32(parsed_line[0]) == -1):
            break

        # the element type
        element_type = np.int32(parsed_line[2])

        # the component number
        component_number = np.int32(parsed_line[3])

        # the number of nodes
        num_nodes = get_num_nodes_ele(element_type)

        # connectivity
        c.append(num_nodes)
        for i in range(num_nodes):
            c.append(np.int32(parsed_line[4+i]))

        comp_names.append(component_number)
        element_codes.append(element_type)

    # convert c to numpy array
    c = np.array(c, dtype=np.int32)

    return c, comp_names, element_codes


def get_num_nodes_ele(ele_code):
    '''Given the code in the ele file, return the
    number of nodes of the element.

    :param ele_code:
        The ele code. See tech_doc. page 36/37.

    :return:
        The number of nodes.
    '''

    if ele_code == 65:
        # T6 element
        return 6
    elif ele_code == 66:
        # Q8 element
        return 8
    else:
        print('code {} unknown!'.format(ele_code))