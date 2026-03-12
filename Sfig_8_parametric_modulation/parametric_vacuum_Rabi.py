import h5py
import numpy as np
import pylab as plt
from matplotlib.colors import LinearSegmentedColormap

# Load data
with h5py.File("Valadares-Dorogov-FAR/Sfig_8_parametric_modulation/data/20250814_191907_lakeside_parametric_vacuumrabi.h5", "r") as f:
    z = np.array(f["data"])  # z.shape should be (len(y), len(x))
    x = np.array(f["flux_frequency"])  # shift center to 0
    y = np.array(f["flux_duration"]) * 4  # convert to nanoseconds

# Create meshgrid for plotting
X, Y = np.meshgrid(x, y)

# Renormalize Z to [0, 1]
z_min, z_max = np.min(z), np.max(z)
z_norm = (z - z_min) / (z_max - z_min)

# Define custom colormap: blue-white-red
color1 = "#1f53b4" 
color2 = "#c01616" 
custom_cmap = LinearSegmentedColormap.from_list(
    "blue_white_red", [color1, "white", color2]
)

# Plot
plt.figure(figsize=(5 * 0.8, 4 * 0.8))  # specific size
pcm = plt.pcolormesh(X / 1e6, Y, z_norm, shading="auto", cmap=custom_cmap)
plt.xlabel("Flux modulation frequency (MHz)")
plt.ylabel("Interaction length (ns)")
# plt.title('Vacuum Rabi Signal')

# Colorbar
plt.colorbar(pcm, label=r"$|1\rangle$ population", ticks=[0, 1])

# Save as EMF before plotting
plt.yticks([250, 500, 750, 1000, 1250, 1500])
plt.tight_layout()
# plt.savefig("vacuum_rabi_plot.jpeg", format="jpeg", dpi=300)

# Show plot
plt.show()
