import numpy as np
import qutip
import matplotlib.pyplot as plt
from tools.hamiltonian import build_JC_hamiltonian, eigenvec, E_list, build_decoupling_matrix
from tools.drive_envelope import build_drive_envelope
from pathlib import Path
import time
start = time.perf_counter()


H = build_JC_hamiltonian()
cdim_0 = H.dims[0][0]
qdim_0 =  H.dims[0][1] # assumed to be 2

# Qubit T1 and T2
T1 = np.inf#10000
T2_list = [2100]#, 2200, 2300, 2400, 2500, 2600, 2700, 2800]
# T2 = np.inf# 29999

for T2 in T2_list:
    Tphi = np.inf#1 / (1 / T2 - 0.5 / T1)
    cavT1 = 500_000

    # Cavity operators
    a  = qutip.tensor(qutip.destroy(cdim_0), qutip.qeye(qdim_0))  
    ad = a.dag()   

    # Qubit operators (2-dimensional Hilbert space)
    sp = qutip.tensor(qutip.qeye(cdim_0), qutip.sigmap())     # Pauli +
    sm = qutip.tensor(qutip.qeye(cdim_0), qutip.sigmam())     # Pauli -
    sx = qutip.tensor(qutip.qeye(cdim_0), qutip.sigmax())     # Pauli X
    sz = qutip.tensor(qutip.qeye(cdim_0), qutip.sigmaz())     # Pauli Z

    # Drive sequence
    state_name = "0p3"
    s_0to1p = 68
    s_1pto2m = 48
    s_2mto3p = 44
    drive_0to1p_params = (0.00442545*0.5/2, E_list[2] - E_list[0], 0, s_0to1p)
    drive_1pto2m_params = (0.00888164/2, E_list[3] - E_list[2], s_0to1p * 4, s_1pto2m)
    drive_2mto3p_params = (0.00954657/2, E_list[6] - E_list[3], (s_0to1p + s_1pto2m) * 4, s_2mto3p)
    t_max = (s_0to1p + s_1pto2m + s_2mto3p)*4

    # Full time-dependent Hamiltonian
    Hcoeff1, Hcoeff1_conj = build_drive_envelope(*drive_0to1p_params)
    Hcoeff2, Hcoeff2_conj  = build_drive_envelope(*drive_1pto2m_params)
    Hcoeff3, Hcoeff3_conj  = build_drive_envelope(*drive_2mto3p_params)
    Hd1 = sm # rising operator (not a typo)
    Hd2 = sp # lowering operator (not a typo)
    H_td = [H, 
            [Hd1, Hcoeff1], 
            [Hd2, Hcoeff1_conj], 
            [Hd1, Hcoeff2], 
            [Hd2, Hcoeff2_conj], 
            [Hd1, Hcoeff3], 
            [Hd2, Hcoeff3_conj], ]

    # Build projector list
    eigenvec_list = [qutip.tensor(qutip.basis(cdim_0, 0), qutip.basis(2, 0))]
    eigenvec_list += [eigenvec(ii, 2*jj-1) for ii in range(1, 6) for jj in range(2)]
    proj_list = [qutip.ket2dm(evec) for evec in eigenvec_list] 

    # Build jump operators
    c_T1 = np.sqrt(1/T1)*sp
    c_Tphi = np.sqrt((1/Tphi) / 2.0) * -sz
    c_cavT1 = np.sqrt(1/cavT1)*a
    c_ops = [c_T1, c_Tphi, c_cavT1]

    # Solve master equation
    psi0 = qutip.tensor(qutip.basis(cdim_0, 0), qutip.basis(qdim_0, 0))
    tlist = np.linspace(0.0, t_max, 1001)
    result = qutip.mesolve(H_td, psi0, tlist, c_ops, 
                        proj_list,
                        options = qutip.Options(store_states=True))

    # Get final state
    ADM = build_decoupling_matrix() 
    rho_JC_final = result.states[-1]# qutip.ket2dm(psi_JC_final)
    rho_dec_final = ADM*rho_JC_final*ADM.dag()
    rho = rho_dec_final

    # Save final state
    dims_str = "x".join(str(d) for d in rho.dims[0])  # e.g. "30x3"
    fname = f"rho_{state_name}_dims{dims_str}_T1{T1}_T2{T2}_cavT1{cavT1}.npz"
    np.savez(
        Path("data//states") / fname,
        rho_data=rho.full(),     
        rho_dims=rho.dims,       
        state_name=state_name,
        T1=float(T1),
        T2=float(T2),
        cavT1=float(cavT1),
    )
    print(f"Saved: {fname}")
    end = time.perf_counter()
    print(f"Execution time: {end - start:.3f} s")

    ## Print state evolution
    plt.figure()
    for i in range(len(proj_list)):
        plt.plot(tlist, result.expect[i], label = i)
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot Wigner
    rho_cav_final = rho_dec_final.ptrace(0)

    # Phase-space grid
    x = np.linspace(-5, 5, 200)
    p = np.linspace(-5, 5, 200)
    # Compute Wigner function
    W = qutip.wigner(rho_cav_final, x, p)
    # Plot
    plt.figure(figsize=(6, 5))
    plt.contourf(x, p, W, 100, cmap="RdBu_r",vmin = -1/np.pi, vmax = 1/np.pi)
    plt.xlabel("x")
    plt.ylabel("p")
    plt.title("Wigner function")
    plt.colorbar()
    plt.show()

    # Calculate fidelity to target state
    p_off = qutip.basis(cdim_0, 3)*qutip.basis(cdim_0, 0).dag() # Upper diagonal
    phase = np.angle((rho_cav_final*p_off).tr())
    psi_target = (qutip.basis(cdim_0, 0) + np.exp(-1j*phase)*qutip.basis(cdim_0, 3)).unit()
    rho_target = qutip.ket2dm(psi_target)

    print(rho_cav_final)
    print("Cavity fidelity: ", qutip.fidelity(rho_target, rho_cav_final)**2)
    # Compute Wigner function
    W = qutip.wigner(rho_target, x, p)
    # Plot
    plt.figure(figsize=(6, 5))
    plt.contourf(x, p, W, 100, cmap="RdBu_r",vmin = -1/np.pi, vmax = 1/np.pi)
    plt.xlabel("x")
    plt.ylabel("p")
    plt.title("Wigner function")
    plt.colorbar()
    plt.show()
