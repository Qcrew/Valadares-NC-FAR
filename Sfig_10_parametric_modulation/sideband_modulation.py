import h5py
import numpy as np
import pylab as plt
from matplotlib.colors import LinearSegmentedColormap

# Load data
with h5py.File("Valadares-Dorogov-FAR/Sfig_8_parametric_modulation/data/20250815_113634_lakeside_sideband_modulation.h5", "r") as f:
    z = np.array(f["data"])  # z.shape should be (len(y), len(x))
    x = np.array(f["flux_ampx"])  # shift center to 0
    y = np.array(f["flux_duration"]) * 4  # convert to nanoseconds

def centers_to_edges(v):
    dv = np.diff(v)
    left  = v[0]  - dv[0]/2
    right = v[-1] + dv[-1]/2
    return np.concatenate(([left], (v[:-1]+v[1:])/2, [right]))

xe = centers_to_edges(x)
ye = centers_to_edges(y)

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
print(x.shape)
# Plot
plt.figure(figsize=(5 * 0.8, 4 * 0.8))  # specific size
pcm = plt.pcolormesh(x*0.2/50/10*1e6, y, z_norm.T, shading="auto", cmap=custom_cmap)
plt.xlabel("Modulation amp. (µA)")
plt.ylabel("Interaction length (ns)")
# plt.title('Vacuum Rabi Signal')

# Colorbar
plt.colorbar(pcm, label=r"$|1\rangle$ population", ticks=[0, 1])

# Save as EMF before plotting
plt.yticks([200, 400, 600, 800])
plt.tight_layout()
# plt.savefig("sideband_mod.svg", format="svg", dpi=300)

# Show plot
plt.show()
