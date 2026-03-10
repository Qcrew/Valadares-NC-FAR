
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import h5py
from qutip import *
import os
#current dir for saving the map
# os.chdir('/Users/tanjungkrisnanda/Library/CloudStorage/Dropbox/NTU Grad/Research/Python codes/nus_tomo_23/20250714_Allcoherent')
# os.chdir(r"C:\Users\tanju\Dropbox\NTU Grad\Research\Python codes\nus_tomo_23\20250714_Allcoherent")
from tools.TK_basics import *
import time
start_time = time.time()  # checking how long the code takes


# Dimension number and number of observables (number of displacements)
D = 7
nD = D**2 - 1

# Load inverse map variables for converting pe to rho
data = np.load(Path("data\\maps") / f"map_ideal_D7.npz")
W = data["W"]
beta = data["beta"]

n_obs = beta.shape[0]

# Mapping
C = np.matmul(-W, beta[:, 0])
BETA = np.zeros([nD, n_obs + 1])
BETA[:, 0] = C
BETA[:, 1 : n_obs + 1] = W

# Builds a density matrix from the vector Y
def rho_from_Y(Y_est):
    rho_est = np.zeros([D, D], dtype=np.complex128)
    diagonal = np.append(Y_est[: D - 1], 1 - sum(Y_est[: D - 1]))
    np.fill_diagonal(rho_est, diagonal)  # Populate diagonal of rho

    index_i_list = np.triu_indices(D, 1)[0]
    index_j_list = np.triu_indices(D, 1)[1]
    for k in range(len(index_i_list)):  # Populate off-diagonals of rho
        index_i = index_i_list[k]
        index_j = index_j_list[k]
        rho_est[index_i, index_j] = Y_est[D - 1 + 2 * k] + 1j * Y_est[D + 2 * k]
        rho_est[index_j, index_i] = Y_est[D - 1 + 2 * k] - 1j * Y_est[D + 2 * k]

    return Qobj(rho_est)

# # Experimental observables
state_fname = f"rho_0p2p4_dims6x2_T110000_T21400_cavT1500000.npz"
data = np.load(Path("data\\wigner") / (f"wigner_" + state_fname))
We = data["matrix"]

data = We.reshape(len(We)**2)
X_exp = np.zeros([1 + len(data)])
X_exp[0] = 1
# X_exp[1:] = (Wi.reshape(len(Wi)**2)).real
X_exp[1:] = data

# Estimate the state by applying the inverse map to the experimental data
Y_est = np.zeros(nD)
Y_est = np.matmul(BETA, X_exp)
qRho_est = rho_from_Y(Y_est)  # just a reshaping

#this is maximum likelihood
qRho_est_PSD = PSD_rho(qRho_est.full())


# f1 = fidelity(rho_tar_D, Qobj(qRho_est)) ** 2
# print(f'Fidelity is {np.round(f1,2)}')

print("")
print("--- %s seconds ---" % (time.time() - start_time))


from scipy.optimize import minimize_scalar
cdim = 30 #dimension for simulation
D = 7 #truncation dimension of the reconstructed state
a = destroy(cdim).full()
adag = a.T.conj()

# Get the phases of each superposition coeff
a1 = np.angle(qRho_est_PSD[2, 0])
a2 = np.angle(qRho_est_PSD[4, 0])
a_ref = np.angle(1j)
print(a1, a2, a_ref)

#NOTE the target state definition is below
target_ket = ((1/4)**0.5*basis(D,0) + (3/8)**0.5*np.exp(1j*a1)*basis(D,2) + (3/8)**0.5*np.exp(1j*a2)*basis(D,4)).unit()
# target_ket = ((1/3)**0.5*basis(D,0) + (1/3)**0.5*np.exp(1j*a1)*basis(D,2) + (1/3)**0.5*np.exp(1j*a2)*basis(D,4)).unit()
print(qRho_est_PSD)

def fidelphase(ph):
    cdim = adag.shape[0]
    D = qRho_est_PSD.shape[0]
    qRho = np.zeros([cdim,cdim], dtype=np.complex128)
    tar = np.zeros([cdim,1], dtype=np.complex128)
    #embed the D dimensional state in higher dimension cdim
    #so now we can do simulation of rotation in higher dimension
    qRho[0:D, 0:D] = qRho_est_PSD.full()
    tar[0:D,:] = target_ket.full()

    psi_r = expm(-1j*ph*adag@a)@tar
    f = (psi_r.T.conj()@qRho@psi_r)[0,0].real
    return f

res = minimize_scalar(lambda ph: -fidelphase(ph), bounds=(0, 2*np.pi), method='bounded')

best_ph = res.x

print(f"Optimal phase: {best_ph}")

# now plot we know the target state with phase correction

tar = np.zeros([cdim,1], dtype=np.complex128)
tar[0:D,:] = target_ket.full()
tar_rot = expm(-1j*best_ph*adag@a)@tar
rho = tar_rot@tar_rot.T.conj()

P = expm(1j*np.pi*adag@a)

beta_list = np.linspace(-2.85, 2.85, 41) 
beta_re = beta_list
beta_im = beta_list

# Wi = np.zeros([len(displacements_Q), len(displacements_Q)],dtype =np.complex128)
# for i in np.arange(0, len(displacements_Q)):
#     for j1 in np.arange(0, len(displacements_Q)):
#         beta = beta_re[j1]-1j*beta_im[i]#the minus sign for consistent ordering
#         Da = expm(beta*adag-np.conj(beta)*a)
#         Wi[i,j1] = np.trace(rho@Da@P@Da.T.conj()).real

# X, Y = np.meshgrid(displacements_I, displacements_Q)
# # surf = ax1.plot_surface(X, Y, W.real, cmap=cm.coolwarm, rstride=1, cstride=1, linewidth=0, antialiased=False)#, vmin = -1, vmax = 1)
# # fig.colorbar(surf, shrink=0.5, aspect=5)
# # ax1.set_xlabel('Re'+r'$(\beta)$')
# # ax1.set_ylabel('Im'+r'$(\beta)$')

# create_figure_cm(5,5)
# # plt.pcolormesh(X, Y, W.real, cmap="viridis", shading='auto')#, extent=[AL.min(), AL.max(), AL.min(), AL.max()], origin='lower')
# cf = plt.pcolormesh(
#     displacements_I, displacements_Q, Wi.real, cmap="bwr", vmax=1, vmin=-1
# )
# print(f'min {np.min(Wi)}')
# print(f'max {np.max(Wi)}')

#fidelity between reconstructed state Qobj(qRho_est)
#and the target state with phase correction Qobj(rho[:D,:D])
rho_tar_D = Qobj(rho[:D,:D])
Fid = fidelity(rho_tar_D, Qobj(qRho_est_PSD)) ** 2
print(f'Maximum likelihood fidelity is {Fid}')
#this is Bayesian inference
B_start_time = time.time()
number_of_repetitions = 5792
B_fid_mean, B_fid_std, B_qRho_est = Bysn_rho_v2(2**10, number_of_repetitions*(nD),
                                                rho_tar_D.full(),
                                                qRho_est.full())
# qRho_est = Qobj(qRho_est)
print(f'Bayesian fidelity is {B_fid_mean}')
print(f'Bayesian fidelity error is {B_fid_std}')
print("\nBayesian inference took\n--- %s seconds" % (time.time() - B_start_time))
# print(f'Fidelity is {Fid}')
