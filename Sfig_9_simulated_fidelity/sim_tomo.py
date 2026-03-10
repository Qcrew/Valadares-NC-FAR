import numpy as np
import matplotlib.pyplot as plt
from qutip import *
import time
from tools.calibrate_wigner_pulse import calibrate_pi
from pathlib import Path
start_time = time.time()

############################################################################
######################   Load and prepare state  ###########################
############################################################################

# Retrieve simulated density matrix
fname = "rho_0p3_dims6x2_T115000_T21500_cavT1500000.npz"
# How to read the file
f = np.load(Path("data\\states") / fname, allow_pickle=True)

rho_data = f["rho_data"]
rho_dims = f["rho_dims"].tolist()
cdim_0, qdim_0 = rho_dims[0][0], rho_dims[0][1]

rho = Qobj(rho_data, dims=rho_dims)
state_name = str(f["state_name"])
T1 = int(f["T1"])
T2 = int(f["T2"])
Tphi = 1 / (1 / T2 - 0.5 / T1)
cavT1 = int(f["cavT1"])

print("=== Loaded file ===")
print("Filename:", fname)
print("state_name:", state_name)
print("T1:", T1)
print("T2:", T2)
print("cavT1:", cavT1)
print("rho dims:", rho.dims)

# Increase density matrix dimension
cdim, qdim = 30, 3
V = 0
for k1 in range(cdim_0):
    for k2 in range(qdim_0):
        ket_new = tensor(basis(cdim, k1), basis(qdim, k2))
        ket_old = tensor(basis(cdim_0, k1), basis(qdim_0, k2))
        V += ket_new * ket_old.dag()
V = Qobj(V.full(), dims=[[cdim, qdim], [cdim_0, qdim_0]])
rho = V * rho* V.dag()


############################################################################
#######################   Wigner Tomography  ###############################
############################################################################


# Dispersive Hamiltonian parameters in GHz
chi = 1.62e-3
Kerr = 3e-6*0
alpha = 160e-3

# Wigner parameters
wigner_wait = 272 # Wigner wait time, ideally np.pi/(2*np.pi*chi)
sigma, chop = [8, 4] #for Ry(pi/2), power Rabi 
rescaling = 1. # Wigner of vacuum amplitude

# Operators
ug = basis(qdim, 0)
ue = basis(qdim, 1)
q = destroy(qdim)
qd = q.dag()

Q = tensor(qeye(cdim), destroy(qdim))
C = tensor(destroy(cdim), qeye(qdim))
Cd, Qd = C.dag(), Q.dag()

# Dispersive Hamiltonian
H_dis = -2*np.pi*chi*Cd*C*Qd*Q - 2*np.pi*Kerr/2*Cd*Cd*C*C - 2*np.pi*alpha/2 * Qd*Qd*Q*Q

# Calibrate Wigner pulses
A, pulse = calibrate_pi(sigma, chop, alpha, T1, Tphi, qdim)
print("Finished pulse cal")
Hd = 2*np.pi*A*1j*(Qd - Q)/2 # 1/2 factor for Ry(pi/2)
Hdm = -2*np.pi*A*1j*(Qd - Q)/2 # -1/2 factor for -Ry(pi/2)

H = [H_dis, [Hd, pulse]]
Hm = [H_dis, [Hdm, pulse]]

#jump operators qubit-cavity
c_ops = [
        # Qubit Relaxation
        np.sqrt(1/T1) * Q,
        # Qubit Dephasing
        np.sqrt(2/Tphi) *Qd*Q,#changed
        # Cavity Relaxation
        np.sqrt(1/cavT1) * C,
    ]

tlist = np.linspace(0, sigma*chop, 100) # in ns
tlist2 = np.linspace(0, wigner_wait, 2)

def calc_displaced_parity(beta1, beta2):
    
    Ds = tensor(displace(cdim, beta1 +1j*beta2), qeye(qdim))
    rho_disp = Ds.dag()*rho*Ds

    #first pi/2
    result1 = mesolve(H, rho_disp, tlist, c_ops)

    #dispersive coupling wait time
    result2 = mesolve(H_dis,result1.states[-1], tlist2, c_ops)

    #second pi/2
    result3 = mesolve(H,result2.states[-1], tlist, c_ops)

    #second -pi/2
    result3m = mesolve(Hm,result2.states[-1], tlist, c_ops)

    rho_qt = result3.states[-1].ptrace([1])
    rho_qtm = result3m.states[-1].ptrace([1])
    par_nor = (2*(rho_qt*(ue*ue.dag())).tr()-1).real # parity normal
    par_rev = (2*(rho_qtm*(ue*ue.dag())).tr()-1).real # parity reverse
    #corrected parity
    par_corrected = (par_nor-par_rev)/2

    return par_corrected


# Calculate Wigner over grid
beta_list = np.linspace(-2.85, 2.85, 41) 
W = np.zeros((len(beta_list), len(beta_list)))
for i, beta1 in enumerate(beta_list):
    for j, beta2 in enumerate(beta_list):
        W[i, j] = calc_displaced_parity(beta1, beta2)
        
W /= rescaling


# Save wigner to npz
np.savez(
    Path("data\\wigner") / ("wigner_" + fname),
    # Main data
    matrix=W,
    disp = beta_list,
    state=rho.full(),
    dims=rho.dims,
    # System parameters
    chi=chi,
    Kerr=Kerr,
    alpha=alpha,
    wigner_wait=wigner_wait,
    sigma=sigma,
    chop=chop,
    rescaling=rescaling
)

## Plot results
plt.figure()
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