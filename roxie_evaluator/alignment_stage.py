import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd


class AlignmentStage():
    '''This is a class to model the alignment stages.
    '''

    def __init__(self, parameter_filename):
        '''Default constructor

        :param parameter_filename:
            A parameter filename with the csv file for the stage orientation and
            home position.

        :return:
            Nothing.
        '''

        in_data = pd.read_csv(parameter_filename)
        
        self.E = np.array([in_data['e_x'].values, 
                           in_data['e_y'].values, 
                           in_data['e_z'].values]).T
        
        self.r_0 = in_data['r_0(mm)'].values*1e-3

    def transform_to_global(self, points):
        '''Transform a set of points to global coordinates.

        :param points:
            The points in local coordinates in an (M x 3) numpy array.

        :return:
            The points in global coordinates in an (M x 3) numpy array.
        '''

        r_glob = (self.E @ points.T).T
        r_glob[:, 0] += self.r_0[0]
        r_glob[:, 1] += self.r_0[1]
        r_glob[:, 2] += self.r_0[2]
        
        return r_glob

