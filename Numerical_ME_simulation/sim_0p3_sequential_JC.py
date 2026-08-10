#%%
import numpy as np
import qutip
import pylab as plt

omega_c = 2 * np.pi * 6.868737    # Cavity frequency
omega_01 = 2 * np.pi * 6.868737  
g = 2 * np.pi * 0.012087952       # Coupling strength 
# g = 2 * np.pi * 0.05
alpha = 2 * np.pi * -0.206020329
N_c = 10  # Truncation level for the bosonic mode
tdim = 4
# tdim = 2

a = qutip.tensor(qutip.destroy(N_c), qutip.qeye(tdim))
b = qutip.tensor(qutip.qeye(N_c), qutip.destroy(tdim))
def build_Hamiltonian(omega_c: float, alpha: float, g: float):
    a = qutip.tensor(qutip.destroy(N_c), qutip.qeye(tdim))
    b = qutip.tensor(qutip.qeye(N_c), qutip.destroy(tdim))
    H_c = omega_c * a.dag() * a


    H_q_lin = omega_c * b.dag() * b
    H_q_anharm = 0.5 * alpha * b.dag() * b.dag() * b * b
    H_q = H_q_lin + H_q_anharm

    H_int = g * (a.dag() * b + a * b.dag())

    # Full Lab Frame Static Hamiltonian
    H_0 = H_c + H_q + H_int
    return H_0
H_0 = build_Hamiltonian(omega_c=omega_c, alpha=alpha, g=g)
eigenvalues, eigenvecs = H_0.eigenstates()
E_list = eigenvalues / 2 / np.pi # Eigenenergies of the composite system
T1 = 10_000 # lower bar
T2 = 1_300
# T1 = 15_000 # higher bar
# T2 = 2_800
# T2 = 2_100
# T1 = 8_000
# T2 = 3_800
# T1 = 4_000
# T2 = 3870.6
Tphi = 1 / (1 / T2 - 0.5 / T1)
# T1 = 100_000 # Good coherence
# Tphi = 50_000
# ECD
# T1 = 50_000
# Tphi = 30_000
cavT1 = 500_000
c_T1 = np.sqrt(1/T1) * b
c_Tphi = np.sqrt(2.0 / Tphi) * b.dag() * b
c_cavT1 = np.sqrt(1/cavT1) * a
c_ops = [c_T1, c_Tphi, c_cavT1]
c_ops = [] # check with no decoherence
if tdim == 4:
    resonant_states = np.array([eigenvecs[0], eigenvecs [1], eigenvecs[2], eigenvecs[4], eigenvecs[5], eigenvecs[8], eigenvecs[9], eigenvecs[12], eigenvecs[13]])
elif tdim == 2:
    resonant_states = np.array([eigenvecs[0], eigenvecs [1], eigenvecs[2], eigenvecs[3], eigenvecs[4], eigenvecs[5], eigenvecs[6], eigenvecs[7], eigenvecs[8]])
else:
    raise KeyError("Need to find the correct eigenenergies first")
def build_decoupling_matrix_4_level():
    ## Matrix M transforms resonant JC state into adiabatically decoupled system
    M = 0

    for indx in range(9): # Corresponding detuned state
        if indx == 0:
            v1 = qutip.tensor(qutip.basis(N_c, 0), qutip.basis(tdim, 0))
            v2 = v1
        else:
            n = (indx + 1)//2 # Number of excitations in the state
            sign = -1 if (indx%2 == 1) else 1
            if sign == 1:
                v1 = qutip.tensor(qutip.basis(N_c, n), qutip.basis(tdim, 0))
            if sign == -1:
                v1 = qutip.tensor(qutip.basis(N_c, n-1), qutip.basis(tdim, 1))
                
            v2 = resonant_states[indx]

        M += v1*v2.dag()
    return M

# g = 50 MHz
# s_0to1p = 17.2399560981332 # Default JC + opt coherent error + enhanced g
# s_1pto2m = 12.701587243668424
# s_2mto3p = 12.135164990319023
# p_0to1p = 0.004289764097474519
# p_1pto2m = 0.011285908132770946
# p_2mto3p = 0.02591595478178018
# Experimental values
s_0to1p = 68 # Default JC
s_1pto2m = 48
s_2mto3p = 44
p_0to1p = 0.00442545*0.5/2 
p_1pto2m = 0.00888164/2
p_2mto3p = 0.00954657/2
# Optimised coherent error, 640 ns
# s_0to1p = 67.75 # Default JC + opt coherent error
# s_1pto2m = 45.25
# s_2mto3p = 160 - s_0to1p - s_1pto2m
# p_0to1p = 0.00115592754
# p_1pto2m = 0.004328911336
# p_2mto3p = 0.004987128168
if tdim == 4:
    drive_0to1p_params = (p_0to1p, E_list[2] - E_list[0], 0, s_0to1p) # amp, freq, start time, sigma
    drive_1pto2m_params = (p_1pto2m, E_list[4] - E_list[2], s_0to1p * 4, s_1pto2m)
    drive_2mto3p_params = (p_2mto3p, E_list[9] - E_list[4], (s_0to1p + s_1pto2m) * 4, s_2mto3p)
elif tdim == 2:
    drive_0to1p_params = (p_0to1p, E_list[2] - E_list[0], 0, s_0to1p) # amp, freq, start time, sigma
    drive_1pto2m_params = (p_1pto2m, E_list[3] - E_list[2], s_0to1p * 4, s_1pto2m)
    drive_2mto3p_params = (p_2mto3p, E_list[6] - E_list[3], (s_0to1p + s_1pto2m) * 4, s_2mto3p)
t_max = (s_0to1p + s_1pto2m + s_2mto3p)*4
# Full time-dependent Hamiltonian
def build_drive_envelope(amp, freq, t0, sigma):
    Omega = 2*np.pi * amp # Rabi frequency
    omega = 2*np.pi * freq # pulse frequency
    t_center = t0 + 2*sigma

    def env(t, args):
        if t0 + 4*sigma > t > t0:
            gaussian = np.exp(-0.5 * ((t - t_center) / sigma)**2)
        else:
            gaussian = 0
        return Omega * gaussian * np.exp(-1j*omega * t)
    
    def env_conj(t, args):
        if t0 + 4*sigma > t > t0:
            gaussian = np.exp(-0.5 * ((t - t_center) / sigma)**2)
        else:
            gaussian = 0
        return Omega * gaussian * np.exp(1j*omega * t)
    
    return env, env_conj

Hcoeff1, Hcoeff1_conj = build_drive_envelope(*drive_0to1p_params)
Hcoeff2, Hcoeff2_conj  = build_drive_envelope(*drive_1pto2m_params)
Hcoeff3, Hcoeff3_conj  = build_drive_envelope(*drive_2mto3p_params)
Hd1 = b.dag()
Hd2 = b
H_td = [H_0, 
        [Hd1, Hcoeff1], 
        [Hd2, Hcoeff1_conj], 
        [Hd1, Hcoeff2], 
        [Hd2, Hcoeff2_conj], 
        [Hd1, Hcoeff3], 
        [Hd2, Hcoeff3_conj], ]

psi0 = qutip.tensor(qutip.basis(N_c, 0), qutip.basis(tdim, 0))
tlist = np.linspace(0.0, t_max, 1_001)
# tlist = np.linspace(0.0, t_max+200, 1_301) # with adiabatic detuning
print("The state preparation time is {:.2f} ns\n".format(t_max))
result = qutip.mesolve(H_td, psi0, tlist, c_ops, 
                    options = qutip.Options(store_states=True, nsteps=10_000))

ADM = build_decoupling_matrix_4_level() # project the JC resonant state on the corresponding adiabatically detuned state
if result.states[-1].type == "oper":
    rho_JC_final = result.states[-1]
else:
    rho_JC_final = qutip.ket2dm(result.states[-1])
rho_dec_final = ADM * rho_JC_final * ADM.dag()
rho_cav_final = rho_dec_final.ptrace(0)

p_off = qutip.basis(N_c, 3)*qutip.basis(N_c, 0).dag() # Upper diagonal
phase = np.angle((rho_cav_final*p_off).tr()) # Can extract the single relative phase directly from the corresponding matrix element
psi_target = (qutip.basis(N_c, 0) + np.exp(-1j*phase)*qutip.basis(N_c, 3)).unit()
rho_target = qutip.ket2dm(psi_target)
print("Cavity fidelity: ", qutip.fidelity(rho_target, rho_cav_final)**2)
if tdim == 4:
    print("-"*64)
    f_pop_list = []
    h_pop_list = []
    for ii, inter_state in enumerate(result.states):
        f_pop_list.append(inter_state.ptrace(1)[2,2])
        h_pop_list.append(inter_state.ptrace(1)[3,3])
    max_index_f = np.argmax(f_pop_list)
    max_index_h = np.argmax(h_pop_list)
    print(" The maximum rho_22 is {}e-3\n Time {} ns of {} ns".format(f_pop_list[max_index_f]*1e3, tlist[max_index_f], tlist[-1]))
    print(" The maximum rho_33 is {}e-3\n Time {} ns of {} ns".format(h_pop_list[max_index_h]*1e3, tlist[max_index_h], tlist[-1]))