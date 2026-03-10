import numpy as np
from pathlib import Path

filename = Path("data\\wigner") / "wigner_rho_0p3_dims6x2_T115000_T21500_cavT1500000.npz"

# Load everything
f = np.load(filename, allow_pickle=True)

# Convert to dict so we can rewrite
data = dict(f)

# Rescale the matrix
scale = 0.7891
data["matrix"] = data["matrix"] * scale

# Save back (overwrite file)
np.savez(filename, **data)

print(f"Updated 'matrix' in {filename} (divided by {scale})")