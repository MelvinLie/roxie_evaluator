import numpy as np

def read_output(filename):
    '''Parse the output of a standard ROXIE simulation.

    :param filename:
        The output filename.

    :return:
        The data arrays in a list.
    '''

    # open the file
    file = open(filename, 'r')

    # read it
    Lines = file.readlines()

    # set a counter
    count = 0

    # this is to arm the data collection
    arm = False

    # a list with return data
    ret_data = []

    for line in Lines:
        if (line == '        N     ABSCISSA       ORDINATE\n'):
            arm = True
            continue
        elif(line.split(' ')[-1] == '\n'):
            arm = False
            count = 0
            continue
        
        if arm:
            if count == 0:
                ret_data.append(np.zeros((0, 2)))
            
            x = float(line.split(',')[1])
            y = float(line.split(',')[2][:-1])
            ret_data[-1] = np.append(ret_data[-1], np.array([[x, y]]), axis=0)

            count += 1

    return ret_data