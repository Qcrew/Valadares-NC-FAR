#%%
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from scipy.linalg import expm
from TK_basics import *


def gaussian_2d(xy, A, x0, y0, sigma_x, sigma_y, theta, offset):
    x, y = xy
    x_rot = np.cos(theta) * (x - x0) + np.sin(theta) * (y - y0)
    y_rot = -np.sin(theta) * (x - x0) + np.cos(theta) * (y - y0)
    return A * np.exp(-0.5 * ((x_rot / sigma_x) ** 2 + (y_rot / sigma_y) ** 2)) + offset


def read_hdf5_file(file_path):
    with h5py.File(file_path, "r") as hdf:
        data = hdf["data"]
        cavity_drive_I = hdf["x"]
        cavity_drive_Q = hdf["y"]
        # print([x[0] for x in data[:]])
        return (
            np.array(data[:, :, :]),
            cavity_drive_I[:],
            cavity_drive_Q[:],
        )

# step 1
# data_filename = "/Users/dorogov/env/State_reconstruction/data/20250808_222306_lakeside_rabi_1to4_0.h5"

# step 2
# data_filename = "/Users/dorogov/env/State_reconstruction/data/20250807_211750_lakeside_rabi_1to4_0p25.h5"

# step 3
# data_filename = "/Users/dorogov/env/State_reconstruction/data/20250808_070449_lakeside_rabi_1to4_0p5.h5"

# step 4
# data_filename = "/Users/dorogov/env/State_reconstruction/data/20250808_141436_lakeside_rabi_1to4_1p0.h5"

# |0> + i|2> + |4> state
data_filename = "/Users/dorogov/env/State_reconstruction/data/20250807_154004_lakeside_0p2p4.h5"

(
    data,
    cavity_drive_I,
    cavity_drive_Q,
) = read_hdf5_file(data_filename)

displacement_scale = 1.5

### Fit vacuum to 2D gaussian
vacuum_data = data[:, :, 0] - data[:, :, 1]
print(vacuum_data.shape)
x_grid, y_grid = np.meshgrid(cavity_drive_I, cavity_drive_Q)
xy = np.vstack((x_grid.ravel(), y_grid.ravel()))
initial_guess = [100e-6, 0, 0, 1, 1, 0, 0]

params_opt, covariance = curve_fit(
    lambda xy, amplitude, x0, y0, sigma_x, sigma_y, theta, offset: gaussian_2d(
        xy, amplitude, x0, y0, sigma_x, sigma_y, theta, offset
    ),
    xy,
    vacuum_data.ravel(),
    p0=initial_guess,
)
# Extract the fitted parameters
amplitude_fit, x0_fit, y0_fit, sigma_x_fit, sigma_y_fit, theta_fit, offset_fit = (
    params_opt
)
print(amplitude_fit, x0_fit, y0_fit, sigma_x_fit, sigma_y_fit, theta_fit, offset_fit)
print("parity offset = ", offset_fit)
print("parity rescaling = ", amplitude_fit)
print("vacuum centering = ", (x0_fit, y0_fit))
print("sigmas = ", (sigma_x_fit * displacement_scale, sigma_y_fit * displacement_scale))

parity_offset = offset_fit
parity_rescaling = amplitude_fit

parity_corrected_0 = (data[:, :, 0] - data[:, :, 1] - parity_offset) / parity_rescaling
parity_corrected_1 = (data[:, :, 2] - data[:, :, 3] - parity_offset) / parity_rescaling

#### Reconstruction
DIM = 7
displacements_I = cavity_drive_I * displacement_scale
displacements_Q = cavity_drive_Q * displacement_scale
my_shape = parity_corrected_1.shape

We = parity_corrected_1.T
create_figure_cm(5,5)
# plt.pcolormesh(X, Y, W.real, cmap="viridis", shading='auto')#, extent=[AL.min(), AL.max(), AL.min(), AL.max()], origin='lower')
cf = plt.pcolormesh(
    displacements_I, displacements_Q, We.real, cmap="bwr", vmax=1, vmin=-1
)
print(f'min {np.min(We)}')
print(f'max {np.max(We)}')

