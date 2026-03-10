import numpy as np
import qutip
import matplotlib.pyplot as plt
from tools.hamiltonian import build_JC_hamiltonian, eigenvec, E_list, build_decoupling_matrix, E_res
from tools.drive_envelope import build_drive_envelope
from tools.optimized_drive_envelope import build_opt_drive_envelope
from pathlib import Path
import time
start = time.perf_counter()


H = build_JC_hamiltonian()
cdim_0 = H.dims[0][0]
qdim_0 =  H.dims[0][1] # assumed to be 2

# Qubit T1 and T2
T1 = np.inf
T2_list = [np.inf]
Tphi = np.inf #1 / (1 / T2 - 0.5 / T1)
cavT1 = 500_000

# Cavity operators
a  = qutip.tensor(qutip.destroy(cdim_0), qutip.qeye(qdim_0))  
ad = a.dag()   

# Qubit operators (2-dimensional Hilbert space)
sp = qutip.tensor(qutip.qeye(cdim_0), qutip.sigmap())     # Pauli +
sx = qutip.tensor(qutip.qeye(cdim_0), qutip.sigmax())     # Pauli X
sz = qutip.tensor(qutip.qeye(cdim_0), qutip.sigmaz())     # Pauli Z

# Drive sequence
state_name = "1p"
# state_sequence = [(3, 0), (2, 1), (1, 0), (0, 0), (1, 1)]
state_sequence = [(1, 1), (0,0), (1, 0), (2, 1),  (3, 0)]
state_index_sequence = [2*x[0] - (1 - x[1]) if x[0] > 0 else 0 
                        for x in state_sequence]
eval_sequence = [E_list[x] for x in state_index_sequence]
# transition_freqs_list = np.diff(np.array(eval_sequence)) # GHz
transition_freqs_list = [eval_sequence[0] - eval_sequence[1],
                         eval_sequence[2] - eval_sequence[1],
                         eval_sequence[3] - eval_sequence[2],
                         eval_sequence[4] - eval_sequence[3]]
transition_freqs_list = np.array(transition_freqs_list)

t_max = 600 # Total drive duration
M = 3 # Number of coefficients for each transition
drive_amps_list = np.array([    
    0.047040, 2.908958, -3.122291, -1.067880, -0.170820, 1.276533, 0.462204, -0.246399, -1.013800, -3.676861, 4.588795, -0.941391, 0.500268, -2.527302, 2.640860, -1.116693, 1.996898, -0.605754, -0.163036, 0.429138, -0.063300, 3.723451, -5.161876, 1.060574
]) # Optimized list of drive coefficients (MHz)

# Full time-dependent Hamiltonian

H1 = qutip.tensor(qutip.qeye(cdim_0), qutip.create(qdim_0))
H2 = qutip.tensor(qutip.qeye(cdim_0), qutip.destroy(qdim_0))

# Transfer pulse
t_off = 0
phase_off = 0
H1_coeff_1, H2_coeff_1 = build_opt_drive_envelope(transition_freqs_list, 
                                            drive_amps_list, M, t_max, t_off, phase_off)


# Build projector list
eigenvec_list = [qutip.tensor(qutip.basis(cdim_0, 0), qutip.basis(2, 0))]
eigenvec_list += [eigenvec(ii, 2*jj-1) for ii in range(1, 5) for jj in range(2)]
proj_list = [qutip.ket2dm(evec) for evec in eigenvec_list] 

# Build jump operators
c_T1 = np.sqrt(1/T1)*sp
c_Tphi = np.sqrt( (1/Tphi) / 2.0) * -sz
c_cavT1 = np.sqrt(1/cavT1)*a
c_ops = [c_T1, c_Tphi, c_cavT1]

# Solve master equation
tlist1 = np.linspace(0.0, t_max, 1001)

# First pulse
H_td_1 = [H, [H1, H1_coeff_1], [H2, H2_coeff_1]]
psi0 = eigenvec(1, 1) #qutip.tensor(qutip.basis(cdim_0, 0), qutip.basis(qdim_0, 0))
result = qutip.mesolve(H_td_1, psi0, tlist1, c_ops, 
                    proj_list,
                    options = qutip.Options(store_states=True))

# Get final state
rho_JC_final = result.states[-1]# qutip.ket2dm(psi_JC_final)
target_rho = qutip.ket2dm(eigenvec(3, -1))

print(transition_freqs_list)
print(H)
# print(rho_JC_final)
# print(target_rho)
print(qutip.fidelity(target_rho, rho_JC_final)**2)