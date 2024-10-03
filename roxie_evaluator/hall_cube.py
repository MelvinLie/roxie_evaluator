import numpy as np
import os
import matplotlib.pyplot as plt
import pyvista as pv
import gmsh
import pandas as pd

from .hall_probe import HallProbe

# permeability of free space
mu_0 = 4.0*np.pi*1e-7

class HallCube():
    '''This is a class to model the measurement operation of a 3D Hall cube.
    '''

    def __init__(self, sensor_param_file):
        '''Default constructor

        :param sensor_param_file:
            The file with the sensor parameters

        :return:
            Nothing.
        '''
        
        # read the sensor parameters
        sensor_params_df = pd.read_csv(sensor_param_file)

        # make the three hall sensors
        self.sensor_1 = HallProbe(sensor_params_df['n_1'].values,
                                  sensor_params_df['s(V/T)'].values[0],
                                  sensor_params_df['u0(V)'].values[0],
                                  sensor_params_df['r_1(mm)'].values*1e-3)
        
        self.sensor_2 = HallProbe(sensor_params_df['n_2'].values,
                                  sensor_params_df['s(V/T)'].values[1],
                                  sensor_params_df['u0(V)'].values[1],
                                  sensor_params_df['r_2(mm)'].values*1e-3)

        self.sensor_3 = HallProbe(sensor_params_df['n_3'].values,
                                  sensor_params_df['s(V/T)'].values[2],
                                  sensor_params_df['u0(V)'].values[2],
                                  sensor_params_df['r_3(mm)'].values*1e-3)
        

    def compute_voltages(self, points, evaluator, quad_order=8, strands=(1, 1)):
        '''Compute the Hall voltages based on a field evaluator.

        :param points:
            The evaluation points.

        :param evaluator:
            The evaluator.

        :param quad_order:
            The quadrature order for the numerical integration.

        :param strands:
            The strands tuple (M x N) strands are used for conductors.

        :return:
            The voltages at the observation points in a numpy array.
        '''

        # the return array
        u_ret = np.zeros(points.shape)

        # fill it
        u_ret[:, 0] = self.sensor_1.compute_voltages(points, evaluator, quad_order, strands)
        u_ret[:, 1] = self.sensor_2.compute_voltages(points, evaluator, quad_order, strands)
        u_ret[:, 2] = self.sensor_3.compute_voltages(points, evaluator, quad_order, strands)

        return u_ret
    
    def get_field_component(self, voltages, sensor, orientation='n'):
        '''Compute a certain field component based on measured voltages.
        Notice! The non-orthogonality is not corrected!

        :param voltages:
            The measured voltages in an (M x 3) numpy array.

        :param sensor:
            The sensor number.

        :param orientation:
            The orientation along which You like to evaluate the field.
            Default is 'n', this means we look at the field in the direction
            of the Hall sensors normal vector.
            If 'x', 'y' or 'z' is chosen, the field is divided by the
            corresponding normal vector component. This assumes that the other field
            components are zero.

        :return:
            The B-fields in an (M x 3) numpy array.
        '''

        if sensor == 1:
            b_ret = self.sensor_1.get_field_component(voltages[:, 0], orientation=orientation)
        elif sensor == 2:
            b_ret = self.sensor_2.get_field_component(voltages[:, 1], orientation=orientation)
        elif sensor == 3:
            b_ret = self.sensor_3.get_field_component(voltages[:, 2], orientation=orientation)

        return b_ret