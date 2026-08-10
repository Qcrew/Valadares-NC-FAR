import numpy as np
import pylab as plt
import h5py
import pickle


def get_indices_from_x(my_array: np.ndarray, values: tuple) -> tuple:
    """
    Finds the indices of the array correponding to the set frequency region

    Arguments:
    _____________________________
    my_array (numpy.ndarray): data with the flat regions (Sfig. 8 (c))
    values (tuple): two frequency values [MHz] defining the range where partial averaging and data reduction will be performed later

    Returns:
    _____________________________
    (ind1, ind2) (tuple): the indices of my_array corresponding to the given frequency range
    """
    x_1, x_2 = values
    ind1 = np.argmin(abs(my_array-x_1))
    ind2 = np.argmin(abs(my_array-x_2))
    if ind2 < ind1:
        raise ValueError("Data is not monotonous")
    else:
        return (ind1, ind2)

def average_floor(values: np.ndarray, step: int) -> list:
    """
    Reduces the number of points in the flat regions to make the plot more clear

    Arguments:
    _____________________________
    values (numpy.ndarray): data with the flat regions (Sfig. 8 (c))
    step (int): it will replace batches of step points with their average, effective reducing the number of points step-fold

    Returns:
    _____________________________
    new_values (list): averaged data with less points
    """
    new_values = []
    new_number = len(values) // step
    for ii in range(new_number):
        if ii != (new_number-1):
            new_values.append(np.mean(values[step*ii:(step*ii + step)]))
        else:
            new_values.append(np.mean(values[step*ii:]))
    return new_values

def apply_operation_to_ranges(points: np.ndarray, x_ranges: list, step: int) -> np.ndarray:
    """
    Handles the partial averaging on the dataset

    Arguments:
    _____________________________
    points (numpy.ndarray): data with the flat regions (Sfig. 8 (c))
    x_ranges (list): a list of tuples defining the ranges to apply average_floor
    step (int): it will replace batches of step points with their average, effective reducing the number of points step-fold

    Returns:
    _____________________________
    new_points (np.ndarray): processed data
    """
    chunks = []
    curr_idx = 0
    for start, end in x_ranges:
        if start > curr_idx:
            chunks.append(points[curr_idx:start])
        region_to_process = points[start:end]
        chunks.append(average_floor(region_to_process, step))
        curr_idx = end
    if curr_idx < len(points):
        chunks.append(points[curr_idx:])
    return np.concatenate(chunks)

path = "Valadares-Dorogov-FAR/Sfig_8_parametric_modulation/data/"

# Saving the averaged experimental datasets to pickle files

# filename = "19-16-19_QubitSpec_with_flux_fast.hdf5" # 63 MHz detuning
# filename = "14-13-29_QubitSpec.hdf5" # 90 MHz detuning
# filename = "18-31-43_QubitSpec_with_flux_fast.hdf5" # 117 MHz detuning
# filename = "17-17-42_QubitSpec_with_flux_fast.hdf5" # 232 MHz detuning

# with h5py.File(path+filename, "r") as hdf:
#         print("Keys: %s" % hdf.keys())
#         I_amp = hdf["I"][:, :]
#         freqs = hdf["qubit_frequency"][:]
#         print("Shape of I:", I_amp.shape)
#         print("Shape of freqs:", freqs.shape)
# I_ = np.mean(I_amp[:, :], axis=0)
# freqs = freqs / 1e6

# with open("Valadares-Dorogov-FAR/Sfig_8_parametric_modulation/data/averaged_sidebands_data/sideband_63_MHz.pkl", 'wb') as file:
#     pickle.dump((I_, freqs), file)

# with open("Valadares-Dorogov-FAR/Sfig_8_parametric_modulation/data/averaged_sidebands_data/sideband_90_MHz.pkl", 'wb') as file:
#     pickle.dump((I_, freqs), file)

# with open("Valadares-Dorogov-FAR/Sfig_8_parametric_modulation/data/averaged_sidebands_data/sideband_117_MHz.pkl", 'wb') as file:
#     pickle.dump((I_, freqs), file)

# with open("Valadares-Dorogov-FAR/Sfig_8_parametric_modulation/data/averaged_sidebands_data/sideband_232_MHz.pkl", 'wb') as file:
#     pickle.dump((I_, freqs), file)

# The ranges where the partial averaging is to be done for different detunings (flat regions)
ranges_63_MHz = np.array([(-200, -28), (-5, 25), (63, 95), (123, 160), (183, 200)])
ranges_90_MHz = np.array([(-200, -25), (-5, 67), (90, 160), (182, 200)])
ranges_117_MHz = np.array([(-200, -74), (-55, 42), (63, 160), (180, 200)])
ranges_232_MHz = np.array([(-200, -70), (-45, 162), (182, 200)])

sideband_data = {
    "63 MHz": {"flat regions": ranges_63_MHz, "filename": "sideband_63_MHz.pkl", "colour": "#3E9491"},
    "90 MHz": {"flat regions": ranges_90_MHz, "filename": "sideband_90_MHz.pkl", "colour": "#88b742"},
    "117 MHz": {"flat regions": ranges_117_MHz, "filename": "sideband_117_MHz.pkl", "colour": "#f19f3c"},
    "232 MHz": {"flat regions": ranges_232_MHz, "filename": "sideband_232_MHz.pkl", "colour": "#bc343a"},
}

sideband_to_plot = "117 MHz" # choose which data to plot here

with open("Valadares-Dorogov-FAR/Sfig_8_parametric_modulation/data/averaged_sidebands_data/"+sideband_data[sideband_to_plot]["filename"], 'rb') as file:
    ampss, freqs = pickle.load(file)

x_ranges = [get_indices_from_x(freqs, one_range) for one_range in sideband_data[sideband_to_plot]["flat regions"]]
step = 14

new_freq_av = apply_operation_to_ranges(points=freqs, x_ranges=x_ranges, step=step)
new_amps_av = apply_operation_to_ranges(points=ampss, x_ranges=x_ranges, step=step)

CM = 1 / 2.54
index = 0
fig = plt.figure(figsize=(6.*CM, 2.5*CM))
ax = plt.gca()
ax.plot(new_freq_av, new_amps_av*1e6, color=sideband_data[sideband_to_plot]["colour"], marker="o", markeredgewidth=0.25,
        mfc='none', markersize=1.4, linewidth=0.25)

ax.tick_params(axis='x', labelbottom=False)
ax.tick_params(axis='y', labelleft=False)
fig.savefig("Valadares-Dorogov-FAR/Sfig_8_parametric_modulation/Sideband " + sideband_to_plot + ".pdf", bbox_inches="tight", transparent=True)
