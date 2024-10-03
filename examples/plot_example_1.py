"""Example 6
   ==========

   The M 113 Magnet. a.k.a. the Panzer.
"""

import numpy as np
import os
import pandas as pd
import pyvista as pv
import roxie_evaluator
import matplotlib.pyplot as plt
import time

# %%
# The script parameters
# =====================
directory = 'files\m_113'
filename = 'M113'

# read the measurement positions
meas_pos_pf = pd.read_csv(os.path.join('files\m_113', 'meas_result.csv'))


# these are the measured voltages
y_1 = meas_pos_pf['V_0'].values
y_2 = meas_pos_pf['V_1'].values
y_3 = meas_pos_pf['V_2'].values

# %%
# Make a Hall cube
# ================
hall_cube = roxie_evaluator.HallCube(os.path.join('files\m_113', 'sensor_parameters.csv'))

# %%
# Read the solution
# =================

# initialize an evaluator
evaluator = roxie_evaluator.evaluator(directory, filename)
evaluator.set_symmetry_flags(1, 2, 1)


# %%
# Evaluate the magnetic vector potential
# ======================================
U = hall_cube.compute_voltages(meas_pos_pf.values[:, :3]*1e-3, evaluator)

# %%
# Plot
# ====
pl = pv.Plotter()
pl.set_background('w', top='w')
# pl = evaluator.plot_solution(pl)
pl = evaluator.plot_geometry(pl, iron_color=[0, 224, 0])
pl.add_mesh(meas_pos_pf.values[:, :3]*1e-3, point_size=10.0, render_points_as_spheres=True)
# pl.add_arrows(points, B, color='black', mag=100)
pl.add_axes()
pl.show()

fig = plt.figure()
ax = fig.add_subplot(121)
ax.plot(meas_pos_pf.values[:, 2], U[:, 0], color='C0', label='$U_1$ ROXIE predicted.')
ax.plot(meas_pos_pf.values[:, 2], y_1, '--' ,color='C0', label='$U_1$ measured')
ax.plot(meas_pos_pf.values[:, 2], U[:, 2], color='C1', label='$U_3$ ROXIE predicted')
ax.plot(meas_pos_pf.values[:, 2], y_3, '--' ,color='C1', label='$U_3$ measured')
ax.legend()
ax.set_xlabel('$z$ in mm')
ax.set_title('Hall voltage in V')
ax = fig.add_subplot(122)
ax.plot(meas_pos_pf.values[:, 2], U[:, 1], color='C2', label='$U_2$ ROXIE predicted')
ax.plot(meas_pos_pf.values[:, 2], y_2, '--' ,color='C2', label='$U_2$ measured')
ax.set_xlabel('$z$ in mm')
ax.set_title('Hall voltage in V')
plt.show()

