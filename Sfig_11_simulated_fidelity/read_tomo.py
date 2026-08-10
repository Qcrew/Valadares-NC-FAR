
import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from pathlib import Path

# ---- File to load ----
for T2 in [1500, 2000, 2500, 3000, 3500, 4000, 4500]:
    # # Experimental observables
    filename = f"wigner_rho_1to4_theta0_dims6x2_T110000_T2{T2}_cavT1500000.npz"
    
# filename = "wigner_rho_0p2p4_dims6x2_T110000_T21300_cavT1500000.npz"   # e.g. "wigner_rho_0p3_....npz"

    f = np.load(Path("data//wigner") / filename, allow_pickle=True)
    W = f["matrix"]
    disp = f["disp"]   # beta_list

    rho_data = f["state"]
    rho_dims = f["dims"].tolist()
    rho = qt.Qobj(rho_data, dims=rho_dims)

    # ---- Read parameters ----
    chi = float(f["chi"])
    Kerr = float(f["Kerr"])
    alpha = float(f["alpha"])
    wigner_wait = int(f["wigner_wait"])

    sigma = f["sigma"] 
    chop = f["chop"] 

    rescaling = float(f["rescaling"])

    # ---- Print nicely ----
    print("========== WIGNER DATA ==========")
    print("File:", filename)
    print()

    print("System parameters:")
    print("  chi          =", chi)
    print("  Kerr         =", Kerr)
    print("  alpha        =", alpha)
    print("  wigner_wait  =", wigner_wait)
    print("  sigma, chop  =", sigma, chop)
    print("  rescaling    =", rescaling)
    print()

    print("Quantum state:")
    print("  rho dims     =", rho.dims)
    print("  rho shape    =", rho.shape)
    print("  trace(rho)   =", rho.tr())
    print("  Hermitian    =", rho.isherm)
    print()

    print("Wigner matrix:")
    print("  W shape      =", W.shape)
    print("  W min/max    =", W.min(), W.max())

    print("Displacement grid (beta_list):")
    print(f"  Number of points : {len(disp)}")
    print(f"  Range            : [{disp.min():.3f}, {disp.max():.3f}]")
    print(f"  Step size         : {disp[1]-disp[0]:.4f}")
    print()

    print("================================")
    print(rho.ptrace(0)[:5, :5])
    # Print Wigner
    beta_list = disp
    cf = plt.pcolormesh(
        beta_list,
        beta_list,
        W,
        cmap="bwr",
        vmax=1,
        vmin=-1,
    )

    plt.colorbar(label="W")
    plt.xlabel(r"$\beta_1$")
    plt.ylabel(r"$\beta_2$")
    plt.title("Displaced Parity Map")
    plt.tight_layout()
    plt.show()