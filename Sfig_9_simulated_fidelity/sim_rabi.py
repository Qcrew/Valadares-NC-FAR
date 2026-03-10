import numpy as np
import qutip
import matplotlib.pyplot as plt
from tools.hamiltonian import build_JC_hamiltonian, eigenvec, E_list, build_decoupling_matrix
from tools.drive_envelope import build_drive_envelope
from tools.optimized_drive_envelope import build_opt_drive_envelope
from pathlib import Path
import time
start = time.perf_counter()


H = build_JC_hamiltonian()
cdim_0 = H.dims[0][0]
qdim_0 =  H.dims[0][1] # assumed to be 2

# Qubit T1 and T2
T1 = 15000
T2_list = [2700]
theta_list = [0, 0.5/4, 0.5/2, 0.5]
theta_label_list = ["0", "pi4", "pi2", "pi"]

for indx in range(len(theta_list)):
    for T2 in T2_list:
        gamma_phi = 1/T2 - 1/2/T1   # pure dephasing rate = 1/Tphi
        Tphi = 1 / (1 / T2 - 0.5 / T1)
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
        state_name = "1to4_theta" + theta_label_list[indx]


        t_max = 600 # Total drive duration
        M = 3 # Number of coefficients for each transition
        state_sequence = [(3, 0), (2, 1), (1, 0), (0, 0), (1, 1)]
        state_index_sequence = [2*x[0] - (1 - x[1]) if x[0] > 0 else 0 
                                for x in state_sequence]
        eval_sequence = [E_list[x] for x in state_index_sequence]
        transition_freqs_list = [eval_sequence[0] - eval_sequence[1],
                                eval_sequence[1] - eval_sequence[2],
                                eval_sequence[2] - eval_sequence[3],
                                eval_sequence[4] - eval_sequence[3]]
        transition_freqs_list_3mto1p  = np.array(transition_freqs_list)
        drive_amps_list_3mto1p = np.array([    
            -2.046203, 4.578998, -0.550179, 3.640922, 0.223905, -0.245251, 2.099943, -0.994850, 
            0.633690, 1.097513, 0.138748, 0.875437, 4.575608, -0.007732, -0.622194, 0.221188, 
            -2.241422, -0.630584, -0.366577, 0.402849, -0.132473, -1.806708, 1.105731, 0.539822
        ]) # Optimized list of drive coefficients (MHz)

        transition_freqs_list_1pto3m  = transition_freqs_list_3mto1p[::-1]
        drive_amps_list_1pto3m = np.array([    
            0.047040, 2.908958, -3.122291, -1.067880, -0.170820, 1.276533, 0.462204, -0.246399, -1.013800, -3.676861, 4.588795, -0.941391, 0.500268, -2.527302, 2.640860, -1.116693, 1.996898, -0.605754, -0.163036, 0.429138, -0.063300, 3.723451, -5.161876, 1.060574
        ]) # Optimized list of drive coefficients (MHz)

        H1 = qutip.tensor(qutip.qeye(cdim_0), qutip.create(qdim_0))
        H2 = qutip.tensor(qutip.qeye(cdim_0), qutip.destroy(qdim_0))
        # First transfer pulse
        t_off = 0
        phase_off = 0
        H1_coeff_1, H2_coeff_1 = build_opt_drive_envelope(transition_freqs_list_1pto3m, 
                                                    drive_amps_list_1pto3m, M, t_max, t_off, phase_off)

        # Drive 3mto4p
        s_3mto4p = 44
        theta = theta_list[indx] # Rotation angle in units of 2pi. theta = 0.5 is a full transfer.
        ampx = 2*theta
        drive_3mto4p_params = (0.00938034/2*ampx, E_list[8] - E_list[5], 0, s_3mto4p)

        # Second transfer pulse, played in reverse
        t_off = 0# t_max + 4*s_3mto4p
        phase_off = 0.0 # units of 2pi
        H1_coeff_2, H2_coeff_2 = build_opt_drive_envelope(transition_freqs_list_3mto1p, 
                                                    drive_amps_list_3mto1p, M, t_max, t_off, phase_off)

        # Build projector list
        eigenvec_list = [qutip.tensor(qutip.basis(cdim_0, 0), qutip.basis(2, 0))]
        eigenvec_list += [eigenvec(ii, 2*jj-1) for ii in range(1, 5) for jj in range(2)]
        proj_list = [qutip.ket2dm(evec) for evec in eigenvec_list] 

        # Build jump operators
        c_T1 = np.sqrt(1/T1)*sp
        c_Tphi = np.sqrt(gamma_phi / 2.0) * -sz
        c_cavT1 = np.sqrt(1/cavT1)*a
        c_ops = [c_T1, c_Tphi, c_cavT1]

        # Solve master equation
        tlist1 = np.linspace(0.0, t_max, 1001)
        tlist2 = np.linspace(0.0, s_3mto4p*4, 1001)

        # First pulse
        H_td_1 = [H, [H1, H1_coeff_1], [H2, H2_coeff_1]]
        psi0 = eigenvec(1, 1) #qutip.tensor(qutip.basis(cdim_0, 0), qutip.basis(qdim_0, 0))
        result = qutip.mesolve(H_td_1, psi0, tlist1, c_ops, 
                            proj_list,
                            options = qutip.Options(store_states=True))
        # Rabi drive
        Hcoeff1, Hcoeff1_conj = build_drive_envelope(*drive_3mto4p_params)
        Hd1 = sm # rising operator (not a typo)
        Hd2 = sp # lowering operator (not a typo)
        H_td_2 = [H, 
                [Hd1, Hcoeff1],
                [Hd2, Hcoeff1_conj]] 
        psi0 = result.states[-1]
        result = qutip.mesolve(H_td_2, psi0, tlist2, c_ops, 
                            proj_list,
                            options = qutip.Options(store_states=True))
        # Second pulse
        H_td_3 = [H, [H1, H1_coeff_2], [H2, H2_coeff_2]]
        psi0 = result.states[-1]
        result = qutip.mesolve(H_td_3, psi0, tlist1, c_ops, 
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

        # ## Print state evolution
        # plt.figure()
        # for i in range(9):
        #     plt.plot(tlist1, result.expect[i], label = i)
        # plt.legend()
        # plt.grid(True)
        # plt.show()

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

        # # Calculate fidelity to target state
        # p_off = qutip.basis(cdim_0, 3)*qutip.basis(cdim_0, 0).dag() # Upper diagonal
        # phase = np.angle((rho_cav_final*p_off).tr())
        # psi_target = (qutip.basis(cdim_0, 0) + np.exp(-1j*phase)*qutip.basis(cdim_0, 3)).unit()
        # rho_target = qutip.ket2dm(psi_target)

        # print(rho_cav_final)
        # print("Cavity fidelity: ", qutip.fidelity(rho_target, rho_cav_final)**2)
        # # Compute Wigner function
        # W = qutip.wigner(rho_target, x, p)
        # # Plot
        # plt.figure(figsize=(6, 5))
        # plt.contourf(x, p, W, 100, cmap="RdBu_r",vmin = -1/np.pi, vmax = 1/np.pi)
        # plt.xlabel("x")
        # plt.ylabel("p")
        # plt.title("Wigner function")
        # plt.colorbar()
        # plt.show()
