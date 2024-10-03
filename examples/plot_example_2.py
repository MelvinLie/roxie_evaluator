"""Example 2
   ==========

   Test the MLFMM
"""

import os
import numpy as np
import pandas as pd
import pyvista as pv
import time

import roxie_evaluator
from roxie_evaluator.cpp_mod import compute_A_mlfmm
import matplotlib.pyplot as plt


# %%
# The script parameters
# =====================
directory = 'files\m_113'
filename = 'M113'

# %%
# Read the solution
# =================

# initialize an evaluator
evaluator = roxie_evaluator.evaluator(directory, filename)
evaluator.set_symmetry_flags(1, 2, 1)

num_points = 50

X, Y, Z = np.meshgrid(np.linspace(-2.0, 0.0, num_points),
                      np.linspace(-2.0, 2.0, num_points),
                      np.linspace(-2.0, 2.0, num_points))

points = np.zeros((num_points**3, 3))
points[:, 0] = X.flatten()
points[:, 1] = Y.flatten()
points[:, 2] = Z.flatten()

print('total number of points = {}'.format(num_points**3))

# mask out exterior points
max_z = 0.7
max_x = 1.0
max_y = 0.5
margin = 0.15


mask_x = abs(points[:, 0]) < max_x + margin
mask_y = abs(points[:, 1]) < max_y + margin
mask_z = abs(points[:, 2]) < max_z + margin

mask = mask_x * mask_y * mask_z

points_outer = points[mask == False, :]


print('Computing iron field')

B_iron = evaluator.compute_iron_field_mlfmm(points_outer, field='B')


# plot
pl = pv.Plotter()

pl.set_background('w', top='w')

pl.add_arrows(points_outer, B_iron,
                        mag=1.5,
                         cmap='jet',
                         scalar_bar_args={"title": "|B_coil| in T", "color": 'k'})
pl = evaluator.plot_geometry(pl)
evaluator.plot_coil(pl)
pl.add_axes()

pl.show()
