"""SHiP SC magnet get map template.
   ==========

   Compute the field map for the SC magnet.
"""

import numpy as np
import os
import pandas as pd
import pyvista as pv
import roxie_evaluator
import matplotlib.pyplot as plt

# %%
# Script parameters
# =================

directory = 'files\SHiP\sc_magnet'
filename = 'bemfem'

# threshold for the B field
B_th = 5.0

# %%
# Specity 3D grid
# ===============

nx = 30
ny = 30
nz = 20

eps = 1e-6
X_max = 2.0
Y_max = 2.0
Z_max = 5.0

X, Y, Z = np.meshgrid(np.linspace(eps, X_max, nx),
                      np.linspace(eps, Y_max, ny),
                      np.linspace(eps, Z_max, nz))



# Check the order!!!
points = np.zeros((nx*ny*nz, 3))
points[:, 0] = X.flatten()
points[:, 1] = Y.flatten()
points[:, 2] = Z.flatten()

# %%
# Make the evaluator
# ==================
evaluator = roxie_evaluator.evaluator(directory, filename, cond_filename='roxie_input.opera8')
evaluator.set_symmetry_flags(0, 2, 1)

# %%
# Evaluate the field
# ==================

# compute the B field
B = evaluator.compute_B(points, 10, 10)

B_norm = np.linalg.norm(B, axis=1)

mask = B_norm < B_th

B = B[mask]
points = points[mask, :]



# %%
# Plot the solution in 3D
# =======================
pl = pv.Plotter(shape=(1, 2))
pl.subplot(0, 0)

# evaluator.plot_geometry(pl)
pl.add_arrows(points, B,
                        mag=0.05,
                         cmap='jet',
                         scalar_bar_args={"title": "|B| in T", "color": 'k'})

pl.add_mesh(points, point_size=1.0, render_points_as_spheres=True, color='red')
pl.add_axes()
pl.show_grid()

pl.subplot(0, 1)

evaluator.plot_fem_field(pl, quad_order=8, mag=0.1)
evaluator.plot_iron_feature_edges(pl)
evaluator.plot_coil(pl)
pl.add_axes()
pl.show_grid()
pl.show()