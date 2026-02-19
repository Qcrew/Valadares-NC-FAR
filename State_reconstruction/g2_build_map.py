#%%
import os
os.chdir('/Users/dorogov/env/State_reconstruction/map_storage')
import time
import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
from qutip import *
from TK_basics import *
start_time = time.time()  # checking how long the code takes

D = 7  # dimension to read

my_shape = parity_corrected_1.shape
new_displacements = np.empty(shape=my_shape, dtype=np.complex128)
for ii in range(my_shape[0]):
    for jj in range(my_shape[1]):
        new_displacements[ii, jj] = displacements_I[jj] + -1j * displacements_Q[ii]
#new_displacements arranged such that we start with wigner from top left and go right
#that is also the order of how the observables will be aranged in a vector form later on

AL = new_displacements.reshape(len(new_displacements)**2,1)

n_dis = len(AL) # no of displacement points


# AL = -AL # we applied the reverse in experiments, the map has the same condition number (robustness)
nD = D**2 - 1  # no of parameters for general states
Ntr = D**2  # no of training for obtaining the map, at least D^2

cdim = 50  # truncation for simulation
a = destroy(cdim).full()  # annihilation for cavity
adag = a.T.conj()
P = expm(1j * np.pi * adag @ a) # parity operator 

# displacement operator
def Dis(alpha):
    Di = expm(alpha * adag - np.conj(alpha) * a)
    return Di

# this part is for obtaining the map
X_r = np.zeros([1 + n_dis, Ntr])  # store readouts
X_r[0, :] = np.ones([1, Ntr])  # setting the ones
Y_r = np.zeros([nD, Ntr])  # store the targets
for j in np.arange(0, Ntr):
    # qudit mixed state embedded in the cavity mode
    rd1 = np.zeros([cdim, cdim], dtype=np.complex128)
    u_rand = rand_ket(D)
    r_rand = (u_rand * u_rand.dag()).full()
    rd1[0:D, 0:D] = r_rand  # randRho(D)

    # assign targets
    cw = 0
    # diagonal elements
    for j1 in np.arange(0, D - 1):
        Y_r[cw, j] = rd1[j1, j1].real
        cw += 1
    # off-diagonal elements
    for j1 in np.arange(0, D - 1):
        for j2 in np.arange(j1 + 1, D):
            Y_r[cw, j] = rd1[j1, j2].real
            cw += 1
            Y_r[cw, j] = rd1[j1, j2].imag
            cw += 1

    w = 0
    for v in np.arange(0, n_dis):
        Di = Dis(AL[w])
        rt = Di.T.conj() @ rd1 @ Di
        X_r[w + 1, j] = np.trace(rt @ P).real 
        w += 1

# ridge regression
lamb = 0

# training, now to obtain the map
X_R = np.zeros([1 + nD, Ntr])  # will contain the parameters
X_R[0, :] = np.ones([1, Ntr])  # setting the ones
Y_R = np.zeros([n_dis, Ntr])  # will contain the obs

# re-defining variables
X_R[1 : nD + 1, :] = Y_r
Y_R[:, :] = X_r[1 : n_dis + 1, :]

Error, beta = QN_regression(X_R, Y_R, lamb) # beta here is the process map

M = beta[:, 1 : nD + 1]  
W = np.matmul(np.linalg.inv(np.matmul(np.transpose(M), M)), np.transpose(M))
CN = np.linalg.norm(M, 2) * np.linalg.norm(W, 2)
print(f"Condition number is {CN}")

np.savez(f"map_ideal_D{D}.npz", 
         M = M, W = W, beta = beta, CN = CN)

print("")
print("--- %s seconds ---" % (time.time() - start_time))
