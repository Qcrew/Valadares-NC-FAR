#%%
import pylab as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.ndimage import rotate
import h5py
# from matplotlib.colors import Normalize, LinearSegmentedColormap

def read_hdf5_file(file_path):
    with h5py.File(file_path, "r") as hdf:
        data = hdf["data"]
        cavity_drive_I = hdf["x"]
        cavity_drive_Q = hdf[f"y"]
        # print([x[0] for x in data[:]])
        return (
            np.array(data[:, :, :]),
            cavity_drive_I[:],
            cavity_drive_Q[:],
        )

def gaussian_2d(xy, A, x0, y0, sigma_x, sigma_y, theta, offset):
    x, y = xy
    x_rot = np.cos(theta) * (x - x0) + np.sin(theta) * (y - y0)
    y_rot = -np.sin(theta) * (x - x0) + np.cos(theta) * (y - y0)
    return A * np.exp(-0.5 * ((x_rot / sigma_x) ** 2 + (y_rot / sigma_y) ** 2)) + offset

    
def make_the_plot(filename: str):
    (
        data,
        cavity_drive_I,
        cavity_drive_Q,
    ) = read_hdf5_file(filename)
    displacement_scale = 1.5
    ### Fit vacuum to 2D gaussian
    vacuum_data = data[:, :, 0] - data[:, :, 1]
    x_grid, y_grid = np.meshgrid(cavity_drive_I, cavity_drive_Q)
    xy = np.vstack((x_grid.ravel(), y_grid.ravel()))
    initial_guess = [100e-6, 0, 0, 1, 1, 0, 0]
    params_opt, _ = curve_fit(
        lambda xy, amplitude, x0, y0, sigma_x, sigma_y, theta, offset: gaussian_2d(
            xy, amplitude, x0, y0, sigma_x, sigma_y, theta, offset
        ),
        xy,
        vacuum_data.ravel(),
        p0=initial_guess,
    )
    amplitude_fit, x0_fit, y0_fit, sigma_x_fit, sigma_y_fit, theta_fit, offset_fit = (params_opt)
    parity_offset = offset_fit
    parity_rescaling = amplitude_fit
    # parity_corrected_0 = (data[:, :, 0] - data[:, :, 1] - parity_offset) / parity_rescaling
    parity_corrected_1 = (data[:, :, 2] - data[:, :, 3] - parity_offset) / parity_rescaling
    #### Reconstruction
    DIM = 7
    displacements_I = cavity_drive_I * displacement_scale
    displacements_Q = cavity_drive_Q * displacement_scale
    my_shape = parity_corrected_1.shape
    new_displacements = np.empty(shape=my_shape, dtype=np.complex128)
    for ii in range(my_shape[0]):
        for jj in range(my_shape[1]):
            new_displacements[ii, jj] = displacements_I[ii] + 1j * displacements_Q[jj]
    # flat_displacements = new_displacements.flatten()
    # flat_parities = parity_corrected_1.flatten()
    # unprocessed_dm = reconstruct_wigner(flat_parities, flat_displacements, DIM)
    # rho = qutip.Qobj(unprocessed_dm).unit()
    # rotation_angle_14 = np.angle(rho.full()[1, 4]) / 2 / np.pi
    # rho_MLE = PSD_MLE_rho(rho=unprocessed_dm).unit()
    fig = plt.figure(figsize=(10, 8))
    plt.xlabel("I displacement")
    plt.ylabel("Q displacement")
    my_x = cavity_drive_I * displacement_scale
    my_y = cavity_drive_Q * displacement_scale
    my_z = parity_corrected_1.T
    cf = plt.pcolormesh(
        my_x,
        my_y,
        my_z,
        cmap="bwr",
        vmax=1,
        vmin=-1,
    )
    plt.title(
        f"Max = {np.max(parity_corrected_1):.3f}, min = {np.min(parity_corrected_1):.3f}"
    )
    plt.xticks(ticks=[-2, 0, 2])
    plt.yticks(ticks=[-2, 0, 2])
    plt.gca().set_aspect("equal")
    return (fig, cf, my_x, my_y, my_z)

#%% Plot the Wigner tomography results
path = "data/"
filename = "20250807_154004_lakeside_0p2p4.h5"
fig, cf, x_, y_, z_ = make_the_plot(filename=path+filename)

#%% Rotate the plot
angle = -64.98
rotated_z_ = rotate(z_, angle, reshape=False, order=3,
                          mode="constant", cval=0.6, prefilter=True)
cut_data = rotated_z_[:, len(x_)//2]

#%% Plot the rotated data
fig, axes = plt.subplots(3, 1, figsize=(12, 6))

axes[0].imshow(z_, origin='lower', cmap='PiYG')
axes[0].set(title="Original Data", xlabel=r"Re[$\beta]", ylabel=r"Im[$\beta]")
# plt.colorbar()

axes[1].imshow(rotated_z_, origin='lower', cmap='PiYG')
axes[1].set(title="Original Data", xlabel=r"Re[$\beta]", ylabel=r"Im[$\beta]")
# plt.title('Rotated Data')
# plt.colorbar()
# plt.xlabel(r'Re[$\beta]')
# plt.ylabel(r'Im[$\beta]')

plt.subplot(1, 3, 3)
plt.plot(cut_data)
plt.title(r'Cut at Re[$\beta$]=0 after {}° rotation'.format(angle))
plt.xlabel(r'Re[$\beta$] (after rotation)')
plt.ylabel(r'Im[$\beta] (after rotation)')

plt.tight_layout()
plt.show()

#%% Plotting the 1-D cut
CM = 1 / 2.54
FONTSIZE = 11
colour_1D_Wigner = "#b42231"
fig = plt.figure(figsize=(2.292634815*CM, 1*CM))
ax = plt.gca()
ax.plot(x_, cut_data, color=colour_1D_Wigner, marker="o", markeredgewidth=0.25,
        mfc='none', markersize=2.25, linewidth=0.25)
# ax.set(xlabel="Frequency, MHz", ylabel="Homodyne signal, a.u.")
# plt.show()
ax.set(xlim=[-2.46, 2.46], ylim=[-1.15, 1.15])
ax.tick_params(axis='x', labelbottom=False)
ax.tick_params(axis='y', labelleft=False)
fig.savefig("Figures/Wigner_cut_0+j2+4_new_.pdf", bbox_inches="tight", transparent=True)