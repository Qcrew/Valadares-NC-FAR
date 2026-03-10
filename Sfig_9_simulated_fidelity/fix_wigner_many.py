# import numpy as np
# from pathlib import Path

# folder = Path("data") / "wigner"
# scale = 0.7891

# # Iterate over all .npz files in the folder
# for filename in folder.glob("*.npz"):
#     print(f"Processing {filename}...")

#     # Load file
#     with np.load(filename, allow_pickle=True) as f:
#         data = dict(f)

#     # Check if "matrix" exists
#     if "matrix" not in data:
#         print(f"  -> Skipped (no 'matrix' key)")
#         continue

#     # Rescale
#     data["matrix"] = data["matrix"] * scale

#     # Overwrite file
#     np.savez(filename, **data)

#     print(f"  -> Updated 'matrix' (multiplied by {scale})")

# print("Done.")
