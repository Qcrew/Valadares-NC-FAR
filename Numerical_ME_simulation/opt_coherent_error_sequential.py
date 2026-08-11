#%%
import numpy as np
import qutip
from scipy.optimize import minimize
import time

# --- System Parameters ---
omega_c = 2 * np.pi * 6.868737      
omega_01 = 2 * np.pi * 6.868737
# g_MHz = 50
g_MHz = 12.087952
g = 0.002 * np.pi * g_MHz  
alpha = 2 * np.pi * -0.206020329

ENFORCE_SUM_CONSTRAINT = True
TARGET_SUM_SIGMA = 160.0       # Only used if ENFORCE_SUM_CONSTRAINT is True
MIN_SIGMA = 2.0               # Minimum allowed sigma for any single pulse
MAX_SIGMA = 150.0              # Maximum sigma (used when constraint is False)

N_c = 10  
tdim = 4

target_fidelity = 0.995
class TargetFidelityReached(Exception):
    """Custom exception to halt optimization when target is met."""
    def __init__(self, params, fidelity):
        self.params = params
        self.fidelity = fidelity

a = qutip.tensor(qutip.destroy(N_c), qutip.qeye(tdim))
b = qutip.tensor(qutip.qeye(N_c), qutip.destroy(tdim))
Hd1 = b.dag() 
Hd2 = b 

def build_Hamiltonian(omega_c, alpha, g):
    H_c = omega_c * a.dag() * a
    H_q_lin = omega_c * b.dag() * b
    H_q_anharm = 0.5 * alpha * b.dag() * b.dag() * b * b
    H_q = H_q_lin + H_q_anharm
    H_int = g * (a.dag() * b + a * b.dag())
    return H_c + H_q + H_int

H_0 = build_Hamiltonian(omega_c=omega_c, alpha=alpha, g=g)
eigenvalues, eigenvecs = H_0.eigenstates()
E_list = eigenvalues / 2 / np.pi 

resonant_states = np.array([eigenvecs[0], eigenvecs[1], eigenvecs[2], eigenvecs[4], 
                            eigenvecs[5], eigenvecs[8], eigenvecs[9], eigenvecs[12], eigenvecs[13]])

def build_decoupling_matrix_4_level():
    M = 0
    for indx in range(9): 
        if indx == 0:
            v1 = qutip.tensor(qutip.basis(N_c, 0), qutip.basis(tdim, 0))
            v2 = v1
        else:
            n = (indx + 1) // 2 
            sign = -1 if (indx % 2 == 1) else 1
            if sign == 1:
                v1 = qutip.tensor(qutip.basis(N_c, n), qutip.basis(tdim, 0))
            if sign == -1:
                v1 = qutip.tensor(qutip.basis(N_c, n-1), qutip.basis(tdim, 1))
        v2 = resonant_states[indx]
        M += v1 * v2.dag()
    return M

ADM = build_decoupling_matrix_4_level() 
psi0 = qutip.tensor(qutip.basis(N_c, 0), qutip.basis(tdim, 0))

# --- Time-Dependent Envelope Generator ---
def build_drive_envelope(amp, freq, t0, sigma):
    Omega = 2 * np.pi * amp 
    omega = 2 * np.pi * freq 
    t_center = t0 + 2 * sigma

    def env(t, args):
        if t0 < t < t0 + 4 * sigma:
            gaussian = np.exp(-0.5 * ((t - t_center) / sigma)**2)
        else:
            gaussian = 0.0
        return Omega * gaussian * np.exp(-1j * omega * t)
    
    def env_conj(t, args):
        if t0 < t < t0 + 4 * sigma:
            gaussian = np.exp(-0.5 * ((t - t_center) / sigma)**2)
        else:
            gaussian = 0.0
        return Omega * gaussian * np.exp(1j * omega * t)
    
    return env, env_conj

# --- Objective Function for Optimization ---
iteration = 0

def cost_function(params):
    global iteration
    if ENFORCE_SUM_CONSTRAINT:
        s_0to1p, s_1pto2m, p_0to1p, p_1pto2m, p_2mto3p = params
        s_2mto3p = TARGET_SUM_SIGMA - s_0to1p - s_1pto2m
        
        if s_2mto3p < MIN_SIGMA:
            return 2.0 + abs(s_2mto3p) 
    else:
        s_0to1p, s_1pto2m, s_2mto3p, p_0to1p, p_1pto2m, p_2mto3p = params
    
    # Calculate timings based on current sigmas
    t0_1 = 0.0
    t0_2 = 4 * s_0to1p
    t0_3 = 4 * (s_0to1p + s_1pto2m)
    t_max = 4 * (s_0to1p + s_1pto2m + s_2mto3p)
    
    drive_0to1p_params = (p_0to1p, E_list[2] - E_list[0], t0_1, s_0to1p)
    drive_1pto2m_params = (p_1pto2m, E_list[4] - E_list[2], t0_2, s_1pto2m)
    drive_2mto3p_params = (p_2mto3p, E_list[9] - E_list[4], t0_3, s_2mto3p)
    
    Hcoeff1, Hcoeff1_conj = build_drive_envelope(*drive_0to1p_params)
    Hcoeff2, Hcoeff2_conj = build_drive_envelope(*drive_1pto2m_params)
    Hcoeff3, Hcoeff3_conj = build_drive_envelope(*drive_2mto3p_params)
    
    H_td = [H_0, 
            [Hd1, Hcoeff1], [Hd2, Hcoeff1_conj], 
            [Hd1, Hcoeff2], [Hd2, Hcoeff2_conj], 
            [Hd1, Hcoeff3], [Hd2, Hcoeff3_conj]]

    # Ensure dynamic time list has fine enough resolution regardless of t_max
    tlist = np.linspace(0.0, t_max, int(t_max * 2)) 
    
    # Use sesolve for much faster execution if no collapse operators are active
    result = qutip.sesolve(H_td, psi0, tlist, 
                           options=qutip.Options(store_states=True, nsteps=10_000))
    
    psi_final = result.states[-1]
    rho_JC_final = qutip.ket2dm(psi_final)
    
    rho_dec_final = ADM * rho_JC_final * ADM.dag()
    rho_cav_final = rho_dec_final.ptrace(0)

    p_off = qutip.basis(N_c, 3) * qutip.basis(N_c, 0).dag() 
    phase = np.angle((rho_cav_final * p_off).tr()) 
    psi_target = (qutip.basis(N_c, 0) + np.exp(-1j * phase) * qutip.basis(N_c, 3)).unit()
    rho_target = qutip.ket2dm(psi_target)
    
    fidelity_sq = qutip.fidelity(rho_target, rho_cav_final)**2
    infidelity = 1.0 - fidelity_sq
    
    iteration += 1
    if iteration % 5 == 0:
        print(f"Iter {iteration} | F^2: {fidelity_sq:.5f} | sigmas: {s_0to1p:.2f}, {s_1pto2m:.2f}, {s_2mto3p:.2f} | amps*1e3: {p_0to1p*1e3:.3f}, {p_1pto2m*1e3:.3f}, {p_2mto3p*1e3:.3f}")

    if fidelity_sq >= target_fidelity:
        raise TargetFidelityReached(params, fidelity_sq)
       
    return infidelity

# --- Run Optimization ---
b_amp = (0.0001, 0.03)
if ENFORCE_SUM_CONSTRAINT:
    initial_guess = [68, 48, 0.00442545 * 0.5 / 2, 0.00888164 / 2, 0.00954657 / 2]
    max_allowed_sigma = TARGET_SUM_SIGMA - (2 * MIN_SIGMA)
    b_sigma = (MIN_SIGMA, max_allowed_sigma)   
    bounds = [b_sigma, b_sigma, b_amp, b_amp, b_amp]
    print(f"Starting constrained optimization (Sum of sigmas = {TARGET_SUM_SIGMA})...")
else:
    initial_guess = [68, 48, 44, 0.00442545 * 0.5 / 2, 0.00888164 / 2, 0.00954657 / 2]
    b_sigma = (MIN_SIGMA, MAX_SIGMA)
    bounds = [b_sigma, b_sigma, b_sigma, b_amp, b_amp, b_amp]
    print("Starting unconstrained optimization (Free sigmas)...")

start_time = time.time()
print("Starting optimization...")
try:
    opt_result = minimize(cost_function, initial_guess, 
                          method='Nelder-Mead', 
                          bounds=bounds,
                          options={'xatol': 1e-4, 'fatol': 1e-4, 'maxiter': 500})
    
    if ENFORCE_SUM_CONSTRAINT:
        s1, s2, p1, p2, p3 = opt_result.x
        s3 = TARGET_SUM_SIGMA - s1 - s2
        final_params = [s1, s2, s3, p1, p2, p3]
    else:
        final_params = opt_result.x
        
    final_fid = 1.0 - opt_result.fun
    print("\n--- Optimization Converged (Target Not Reached) ---")

except TargetFidelityReached as success:
    if ENFORCE_SUM_CONSTRAINT:
        s1, s2, p1, p2, p3 = success.params
        s3 = TARGET_SUM_SIGMA - s1 - s2
        final_params = [s1, s2, s3, p1, p2, p3]
    else:
        final_params = success.params
    final_fid = success.fidelity
    print(f"\n--- Optimization Halted: Target Fidelity Reached! ---")

print(f"Final Fidelity^2: {final_fid:.5f}")
print("Optimized Parameters:")
print(f"s_0to1p = {final_params[0]}")
print(f"s_1pto2m = {final_params[1]}")
print(f"s_2mto3p = {final_params[2]}")
print(f"p_0to1p = {final_params[3]}")
print(f"p_1pto2m = {final_params[4]}")
print(f"p_2mto3p = {final_params[5]}")