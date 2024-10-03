import numpy as np

def read_cor_file(filename):
    '''Read the cor file and return the nodes number table.

    :param filename:
        The filename to read.

    :return:
        The node number table. The first column is the node index
        used in the .ele file. The second column is the node index
        used in the hmo file.
    '''

    # open the file
    cor_file = open(filename, 'r')

    # read first line (version info etc)
    line = cor_file.readline()

    # parse the second line
    line = cor_file.readline()
    parsed_line = [s for s in line.split(' ') if s]
    num_nodes = np.int32(parsed_line[0])
    
    # the connectivity table
    table = np.zeros((num_nodes, 2), dtype=np.int32)

    # a counter variable
    cnt = 0

    # now read the actual data
    while (True):
        
        # read this line
        line = cor_file.readline()

        # parse it
        parsed_line = [s for s in line.split(' ') if s]

        # check end
        if(np.int32(parsed_line[0]) == -1):
            break
        
        # fill table
        table[cnt, 0] = np.int32(parsed_line[0])
        table[cnt, 1] = np.int32(parsed_line[1])

        # increment counter
        cnt += 1

    return table