import numpy as np
import qutip
import matplotlib.pyplot as plt

import scipy as sc
from scipy.optimize import curve_fit

erf = sc.special.erf

'''
Define the Hamiltonian
'''
# List of eigenvalues
# 0, 1-, 1+, 2-, 2+, ... 
E_list = np.array([
    0.0000,
    6.8566,
    6.8809,
    13.7211,
    13.7553,
    20.5868,
    20.6627,
    27.4873,
    27.5770,
    34.4016,
    34.4971,
])

# Cavity operators
N = int((len(E_list)+1)/2) # Add ground state and then parse the qubit dimension
a  = qutip.tensor(qutip.destroy(N), qutip.qeye(2))  
ad = a.dag()   

# Qubit operators (2-dimensional Hilbert space)
sx = qutip.tensor(qutip.qeye(N), qutip.sigmax())     # Pauli X
sz = qutip.tensor(qutip.qeye(N), qutip.sigmaz())     # Pauli Z

def eigenvec(n, sign):
    return (qutip.tensor(qutip.basis(N, n), qutip.basis(2, 0)) + sign*qutip.tensor(qutip.basis(N, n-1), qutip.basis(2, 1))).unit()

## Build resonant JC Hamiltonian
H = 0 # Empty Hamiltonian
for indx, E in enumerate(E_list):

    n = (indx + 1)//2 # Number of excitations in the state

    if indx == 0:
        proj = qutip.ket2dm(qutip.tensor(qutip.basis(N, 0), qutip.basis(2, 0))).unit()

    elif indx%2 == 1: # Superposition phase is -
        proj = qutip.ket2dm(eigenvec(n, -1))

    elif indx%2 == 0: # Superposition phase is +
        proj = qutip.ket2dm(eigenvec(n, 1))

    H += 2*np.pi*E*proj

## Build drive Hamiltonian
Hd = sx

'''
Power Rabi parameters
'''

initial_state = eigenvec(3, -1)#qutip.tensor(qutip.basis(N, 0), qutip.basis(2, 0))
final_state = eigenvec(4, 1)
omega = 2*np.pi * (E_list[8] - E_list[5])
sigma, chop = [44, 4]

'''
Power Rabi: Use this to calibrate the amplitude needed to drive a qubit pi pulse
'''
amp = np.linspace(1.0, 2.5, 21)
output = []


#initial guess
A0 = np.sqrt(2/np.pi) / erf(np.sqrt(2))*np.pi/(4*sigma)/2/np.pi
A0 *= 2 # transitions are weakened due to hybridization

for Ax in amp:
    
    A = Ax*A0 # scaling coeff

    def pulse(t, *arg):
        global sigma, chop, omega
        t0 = sigma*chop/2
        g = np.exp( - 1/2 * (t - t0)**2 / sigma**2)
        return 2 * np.pi * A * g* np.cos(omega * t)

    H_td = [H, [Hd, pulse]]

    # psi = basis(2, 0)#initial state
    rhoq = qutip.ket2dm(initial_state)

    tlist = np.linspace(0, sigma*chop, 100)#in ns

    c_ops = []#[
    #     np.sqrt((1 + nbar_qb)/T1)*q,
    #     np.sqrt(nbar_qb/T1)*qd,
    #     np.sqrt(2/Tphi)*qd*q#changed
    # ]

    e_ops = [qutip.ket2dm(final_state),]

    options = qutip.Options(max_step = 1, nsteps = 1e6)

    results = qutip.mesolve(H_td, rhoq, tlist, c_ops= c_ops, e_ops = e_ops, options= options)#, progress_bar = True)

    output += [results.expect[0][-1],]

# for checking
plt.plot(amp, output)
plt.ylabel(r"pe")
plt.xlabel("Amplitude Scale")
plt.title("Power Rabi")
plt.grid()
plt.show()

print(max(output), output.index(max(output)), amp[output.index(max(output))])
A = A0*amp[output.index(max(output))] # this is the correct coeff
print("Total amp: ", A)