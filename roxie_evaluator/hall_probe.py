import numpy as np
import os
import matplotlib.pyplot as plt
import pyvista as pv
import gmsh
import pandas as pd

# permeability of free space
mu_0 = 4.0*np.pi*1e-7

class HallProbe():
    '''This is a class to model the measurement operation of a Hall sensor.
    '''

    def __init__(self, n, s, u0, pos=np.array([0.0, 0.0, 0.0])):
        '''Default constructor

        :param n:
            The orientation vector.

        :param s:
            The sensitivity function.

        :param u0:
            The offset voltage.

        :param pos:
            A position offset to be applied to all observations.
            Default: np.array([0.0, 0.0, 0.0]).

        :return:
            Nothing.
        '''
        
        self.n = n
        self.s = s
        self.u0 = u0
        self.pos = pos

    def compute_voltages(self, points, evaluator, quad_order=8, strands=(1, 1)):
        '''Compute the Hall voltages based on a field evaluator and points.

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

        # copy the sensor points
        sensor_points = points.copy()

        # add the offsets
        sensor_points[:, 0] += self.pos[0]
        sensor_points[:, 1] += self.pos[1]
        sensor_points[:, 2] += self.pos[2]
        
        # evaluate iron field
        B_i = evaluator.compute_iron_field(sensor_points, quad_order=quad_order, field='B')

        # evaluate coil field
        B_c = evaluator.compute_coil_field(sensor_points, strands[0], strands[1], field='B')

        # compute total field
        B = B_i + B_c

        # compute voltages
        return self.s*(self.n[0]*B[:, 0] + self.n[1]*B[:, 1] + self.n[2]*B[:, 2]) + self.u0

    def get_field_component(self, voltages, orientation='n'):
        '''Compute the field component based on measured voltages.
        Notice! This is the field in the direction of the n vector!

        :param voltages:
            The measured voltages in a numpy array.

        :param orientation:
            The orientation along which You like to evaluate the field.
            Default is 'n', this means we look at the field in the direction
            of the Hall sensors normal vector.
            If 'x', 'y' or 'z' is chosen, the field is divided by the
            corresponding normal vector component. This assumes that the other field
            components are zero.
            
        :return:
            The B-fields in a numpy array.
        '''

        # the return array
        b_ret = (voltages - self.u0)/self.s

        if orientation == 'n':
            pass
        elif orientation == 'x':
            b_ret /= self.n[0]
        elif orientation == 'y':
            b_ret /= self.n[1]
        elif orientation == 'z':
            b_ret /= self.n[2]

        return b_ret