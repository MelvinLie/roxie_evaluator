import numpy as np

def read_sol_file(filename, dimension=3, read_b=False, read_j=False):
    '''Read a ROXIE solution file.

    :param filename:
        The filename to read.

    :param dimension:
        The dimension of the problem. Default = 3.

    :param read_b:
        Set this flag to true if You like to read also the B field.

    :param read_j:
        Set this flag to true if You like to also read the J field.

    :return:
        The results for each time step as lists, i.e.:
        The time_list, the node number list node_number_list,
        the list of vector potentials pot_list, the list of vector potential derivatives der_list,
        the list of B fields b_list, the list of b specifyers b_spec the list of J fields, j_list
        the list of J specifyers j_spec.
        The specifyers define the element and the node to which the evaluation belongs to.
    '''

    # open the file
    sol_file = open(filename, 'r')

    # read first line (version info etc)
    line = sol_file.readline()

    # float which is storing the current instance
    time = 0.

    # list of all instances
    time_list = np.array([])

    # containers for all readings at all instances
    pot_list = []
    der_list = []
    j_list = []
    b_list = []

    # list of all potential values
    if dimension == 2:

        # containers for the current readings
        pot_curr = np.zeros((0,1))
        der_curr = np.zeros((0,1))

        # number of values per node
        dofs_per_node = 1

    elif dimension == 3:

        # containers for the current readings
        pot_curr = np.zeros((0,3))
        der_curr = np.zeros((0,3))

        # number of values per node
        dofs_per_node = 3


    # list of the node numbers
    node_number_list = []

    # arm the readings
    arm_pot = False
    arm_j = False
    arm_b = False

    # conter variable
    counter = 0

    while True:

        # Get next line from file
        line = sol_file.readline()

        # if line is empty
        # end of file is reached
        if not line:
            break

        # parse this line
        parsed_line = [s for s in line.split(' ') if s]


        if(np.int32(parsed_line[0]) == -1):

            # this is a new instance
            time_list = np.append(time_list, float(parsed_line[2]))

            print('reading time {} sec.'.format(float(parsed_line[2])))

        elif(np.int32(parsed_line[0]) == -11):

            # arm potential reading
            arm_pot = True

            # allocate space for the current potential readings
            pot_list.append(np.zeros((np.int32(parsed_line[2]), dofs_per_node)))

            # allocate space for the current derivative readings
            der_list.append(np.zeros((np.int32(parsed_line[2]), dofs_per_node)) + np.nan)

            # allocate space for the current node numbers
            node_number_list.append(np.zeros((np.int32(parsed_line[2]),), dtype=np.int32))

            # reset counter
            counter = 0

        elif(np.int32(parsed_line[0]) == -22):

            if read_b:
                # arm flux density reading
                arm_b = True
            

            # allocate space for the flux density readings
            if len(b_list) > 0:

                b_list.append(np.zeros((b_list[-1].shape[0],3)))
                
            else:
                b_list.append(np.zeros((0,3)))
                b_spec = np.zeros((0,2),dtype=np.int32)

            # reset counter
            counter = 0

        elif(np.int32(parsed_line[0]) == -23):

            if read_j:
                # arm current density reading
                arm_j = True
            

            # allocate space for the current potential readings
            if len(j_list) > 0:

                j_list.append(np.zeros((j_list[-1].shape[0],4)))
                
            else:
                j_list.append(np.zeros((0,4)))
                j_spec = np.zeros((0,2),dtype=np.int32)

            # reset counter
            counter = 0

        elif(np.int32(parsed_line[0]) == -99):

            # unarm all
            arm_pot = False
            arm_j = False
            arm_b = False


        elif arm_pot:

            node_number_list[-1][counter] = np.int32(parsed_line[0])

            if dimension == 2:

                pot_list[-1][counter,0] = float(parsed_line[1])

                if len(parsed_line) > 2:
                    der_list[-1][counter,0] = float(parsed_line[2])

            if dimension == 3:

                pot_list[-1][counter,0] = float(parsed_line[1])
                pot_list[-1][counter,1] = float(parsed_line[2])
                pot_list[-1][counter,2] = float(parsed_line[3])

                if len(parsed_line) > 5:
                    der_list[-1][counter,0] = float(parsed_line[5])
                    der_list[-1][counter,1] = float(parsed_line[6])
                    der_list[-1][counter,2] = float(parsed_line[7])


            counter += 1

        elif arm_b:


            if len(b_list) == 1:
                b_list[-1] = np.append(b_list[-1],np.array([[float(parsed_line[2]),float(parsed_line[3]),float(parsed_line[4])]]),axis=0)
                b_spec = np.append(b_spec,np.array([[np.int32(parsed_line[0]),np.int32(parsed_line[1])]],dtype=np.int32),axis=0)
            else:
                b_list[-1][counter] = np.array([float(parsed_line[2]),float(parsed_line[3]),float(parsed_line[4])])

            counter += 1

        elif arm_j:


            if len(j_list) == 1:
                j_list[-1] = np.append(j_list[-1],np.array([[float(parsed_line[2]),float(parsed_line[3]),float(parsed_line[4]),float(parsed_line[5])]]),axis=0)
                j_spec = np.append(j_spec,np.array([[np.int32(parsed_line[0]),np.int32(parsed_line[1])]],dtype=np.int32),axis=0)
            else:
                j_list[-1][counter] = np.array([float(parsed_line[2]),float(parsed_line[3]),float(parsed_line[4]),float(parsed_line[5])])

            counter += 1

    sol_file.close()

    if read_b and read_j:
        return time_list, node_number_list, pot_list, der_list, b_list, b_spec, j_list, j_spec

    elif read_b:
        return time_list, node_number_list, pot_list, der_list, b_list, b_spec

    elif read_j:
        return time_list, node_number_list, pot_list, der_list, j_list, j_spec

    else:
        return time_list, node_number_list, pot_list, der_list
    

def read_field_solution(filename, field):
    '''Read a ROXIE solution file for the magnetic flux density B.
    ML: NOT DONE YET!

    :param filename:
        The filename to read.

    :param field:
        A string specifying the field You like to read. Options are:
        'B': B field
        'H': H field
    
    :return:
        A numpy array of dimension M x 3 where M is the number of finite
        elements times the number of nodes per element.
        Another numpy array of integ 
    '''

    if field == 'B':
        field_spec = -22
    elif field == 'H':
        field_spec = -24
    else:
        print('Field {} unknown!'.format(field))
        return None
    
    # open the file
    sol_file = open(filename, 'r')

    # read first line (version info etc)
    line = sol_file.readline()

    # containers for all readings at all instances
    fields_list = []

    # a list of times
    time_list = []

    # containers for the current readings
    fields_curr = np.zeros((0,3))

    # list of the node numbers
    node_number_list = []

    # conter variable
    counter = 0

    arm_read = False

    while True:

        # Get next line from file
        line = sol_file.readline()

        # if line is empty
        # end of file is reached
        if not line:
            break

        # parse this line
        parsed_line = [s for s in line.split(' ') if s]


        if(np.int32(parsed_line[0]) == -1):

            # this is a new instance
            time_list = np.append(time_list, float(parsed_line[2]))

            print('reading time {} sec.'.format(float(parsed_line[2])))

        elif(np.int32(parsed_line[0]) == field_spec):

            # arm reading
            arm_read = True

            # allocate space for the current potential readings
            fields_list.append(np.zeros((np.int32(parsed_line[2]), 3)))

            # reset counter
            counter = 0

        elif(np.int32(parsed_line[0]) == -99):

            # unarm all
            arm_read = False


        elif arm_read:

            if len(fields_list) == 1:
                fields_list[-1] = np.append(fields_list[-1], np.array([[float(parsed_line[2]), float(parsed_line[3]), float(parsed_line[4])]]), axis=0)
                b_spec = np.append(b_spec, np.array([[np.int32(parsed_line[0]), np.int32(parsed_line[1])]], dtype=np.int32), axis=0)
            else:
                fields_list[-1][counter] = np.array([float(parsed_line[2]), float(parsed_line[3]), float(parsed_line[4])])

            counter += 1


    sol_file.close()


def sort_solution(node_num, sol):
    '''Sort a solution vector such that its elements correspont to
    the nodes of the finite element mesh.

    :param node_num:
        The node numbers for each entry of the solution vector.

    :param sol:
        The solution vector.

    :return:
        The sorted solution vector.
    '''
    node_num_table = np.zeros((node_num.shape[0], 2), dtype=np.int32)
    node_num_table[:, 0] = np.linspace(0, node_num.shape[0]-1, node_num.shape[0])
    node_num_table[:, 1] = node_num
    indx_sort = np.argsort(node_num)

    return sol[node_num_table[indx_sort, 0], :]