import numpy as np
import qutip
import scipy as sc


'''
Define the JC Hamiltonian from experimental params
'''

# List of eigenvalues
# 0, 1-, 1+, 2-, 2+, ... 
# E_list = np.array([ # experimental model
#     0.0000, # 0 
#     6.8566, # 1m
#     6.8809, # 1p
#     13.7211, # 2m
#     13.7553, # 2p
#     20.5868, # 3m
#     20.6285, # 3p
#     27.4531, # 4m
#     27.5010, # 4p
#     34.3166, #34.4016, Theory
#     34.3709, #34.4971, Theory
# ])
E_list = [ # optimization model
    0.0000, # 0
    6.8566, # 1m
    6.8809, # 1p
    13.7203, # 2m
    13.7547, # 2p
    20.5852, # 3m
    20.6273, # 3p
    27.4507, # 4m
    27.4993, # 4p
    34.3166, # 5m
    34.3709, # 5p
    # 41.1827, # 6m
    # 41.2423, # 6p
]

RWA_list = [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
E_res = (E_list[2] + E_list[1])/2
E_list = [E_list[x] - E_res*RWA_list[x] 
          for x in range(len(RWA_list))] # apply RWA

# Cavity operators
N = int((len(E_list)+1)/2) # Add ground state and then parse the qubit dimension

def eigenvec(n, sign):
    return (qutip.tensor(qutip.basis(N, n), qutip.basis(2, 0)) + sign*qutip.tensor(qutip.basis(N, n-1), qutip.basis(2, 1))).unit()

def build_JC_hamiltonian():
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
    return H

def build_decoupling_matrix():
    ## Matrix M transforms resonant JC state into adiabatically decoupled system
    M = 0

    for indx in range(len(E_list)): # Corresponding detuned state

        if indx == 0:
            v1 = qutip.tensor(qutip.basis(N, 0), qutip.basis(2, 0))
            v2 = v1
        else:
            n = (indx + 1)//2 # Number of excitations in the state
            sign = -1 if (indx%2 == 1) else 1
            if sign == 1:
                v1 = qutip.tensor(qutip.basis(N, n), qutip.basis(2, 0))
            if sign == -1:
                v1 = qutip.tensor(qutip.basis(N, n-1), qutip.basis(2, 1))
                
            v2 = eigenvec(n, sign)

        # print(v1*v2.dag())
        M += v1*v2.dag()
    
    return M
