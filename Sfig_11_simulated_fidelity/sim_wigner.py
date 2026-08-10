import numpy as np
import matplotlib.pyplot as plt
from qutip import *
import time
from tools.calibrate_wigner_pulse import calibrate_pi
from multiprocessing import Pool, cpu_count
from pathlib import Path
start = time.time()

G = {}  # store common objects here
def init_worker(common):
    global G
    G = common

def prepare_state(fname, cdim, qdim):
    # Retrieve simulated density matrix
    # How to read the file
    f = np.load(fname, allow_pickle=True)

    rho_data = f["rho_data"]
    rho_dims = f["rho_dims"].tolist()
    cdim_0, qdim_0 = rho_dims[0][0], rho_dims[0][1]

    rho = Qobj(rho_data, dims=rho_dims)
    state_name = str(f["state_name"])
    T1 = f["T1"]
    T2 = f["T2"]
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
    V = 0
    for k1 in range(cdim_0):
        for k2 in range(qdim_0):
            ket_new = tensor(basis(cdim, k1), basis(qdim, k2))
            ket_old = tensor(basis(cdim_0, k1), basis(qdim_0, k2))
            V += ket_new * ket_old.dag()
    V = Qobj(V.full(), dims=[[cdim, qdim], [cdim_0, qdim_0]])
    rho = V * rho* V.dag()

    return {"rho": rho, 
            "T1": T1, 
            "T2": T2, 
            "Tphi": Tphi, 
            "cavT1": cavT1}

def prepare_wigner_operators(wigner_params_0, state_params, cdim, qdim):

    chi = wigner_params_0["chi"]
    Kerr = wigner_params_0["Kerr"]
    alpha = wigner_params_0["alpha"]
    sigma = wigner_params_0["sigma"]
    chop = wigner_params_0["chop"]

    T1 = state_params["T1"]
    cavT1 = state_params["cavT1"]
    Tphi = state_params["Tphi"]

    # Operators
    q = destroy(qdim)
    qd = q.dag()

    Q = tensor(qeye(cdim), destroy(qdim))
    C = tensor(destroy(cdim), qeye(qdim))
    Cd, Qd = C.dag(), Q.dag()

    # Dispersive Hamiltonian
    H_dis = -2*np.pi*chi*Cd*C*Qd*Q - 2*np.pi*Kerr/2*Cd*Cd*C*C - 2*np.pi*alpha/2 * Qd*Qd*Q*Q

    # Calibrate Wigner pulses
    A, _ = calibrate_pi(sigma, chop, alpha, T1, Tphi, qdim)
    print("Finished pulse cal")
    Hd = 2*np.pi*A*1j*(Qd - Q)/2 # 1/2 factor for Ry(pi/2)
    Hdm = -2*np.pi*A*1j*(Qd - Q)/2 # -1/2 factor for -Ry(pi/2)

    #jump operators qubit-cavity
    c_ops = [
            # Qubit Relaxation
            np.sqrt(1/T1) * Q,
            # Qubit Dephasing
            np.sqrt(2/Tphi) *Qd*Q,#changed
            # Cavity Relaxation
            np.sqrt(1/cavT1) * C,
        ]

    ops = {
        # Hamiltonians
        "H_dis": H_dis,
        "Hd": Hd, 
        "Hdm": Hdm, 
        # operators
        "c_ops" : c_ops,
    }

    return wigner_params_0 | ops

def run_simulation(beta_pair):
    
    beta1, beta2 = beta_pair
    rho = G["rho"]
    cdim, qdim = rho.dims[0][0], rho.dims[0][1]
    sigma = G["sigma"]
    chop = G["chop"]
    wigner_wait = G["wigner_wait"]
    H_dis, Hd, Hdm = G["H_dis"], G["Hd"], G["Hdm"]
    c_ops = G["c_ops"]

    def pulse(t, *arg):
        t0 = sigma*chop/2
        g = np.exp( - 1/2 * (t - t0)**2 / sigma**2)
        return g
    
    H = [H_dis, [Hd, pulse]]
    Hm = [H_dis, [Hdm, pulse]]
    ue = basis(qdim, 1)

    tlist = np.linspace(0, sigma*chop, 100) # in ns
    tlist2 = np.linspace(0, wigner_wait, 2)

    Ds = tensor(displace(cdim, beta1 +1j*beta2), qeye(qdim))
    rho_disp = Ds.dag()*rho*Ds

    #first pi/2
    result1 = mesolve(H, rho_disp, tlist, c_ops)

    #dispersive coupling wait time
    result2 = mesolve(H_dis, result1.states[-1], tlist2, c_ops)

    #second pi/2
    result3 = mesolve(H, result2.states[-1], tlist, c_ops)

    #second -pi/2
    result3m = mesolve(Hm, result2.states[-1], tlist, c_ops)

    rho_qt = result3.states[-1].ptrace([1])
    rho_qtm = result3m.states[-1].ptrace([1])
    par_nor = (2*(rho_qt*(ue*ue.dag())) .tr()-1).real # parity normal
    par_rev = (2*(rho_qtm*(ue*ue.dag())).tr()-1).real # parity reverse
    #corrected parity
    par_corrected = (par_nor-par_rev)/2

    return par_corrected

if __name__ == "__main__":
    
    T1 = 15000
    T2 = 2700
    theta_list = [0, 0.5/4, 0.5/2, 0.5]
    theta_label_list = ["0", "pi4", "pi2", "pi"]
    
    for indx in range(len(theta_list)):
        fname = f"rho_1to4_theta" + theta_label_list[indx] + f"_dims6x2_T1{T1}_T2{T2}_cavT1500000.npz"
        print(fname)
        cdim, qdim = 30, 3
        state_params = prepare_state(Path("data\\states") / fname, cdim, qdim)
        wigner_params_0 ={
            # Dispersive Hamiltonian parameters in GHz
            "chi" : 1.62e-3,
            "Kerr" : 3e-6*0,
            "alpha" : 160e-3,
            # Wigner parameters
            "wigner_wait" : 272, # Wigner wait time, ideally np.pi/(2*np.pi*chi)
            "sigma" : 8, 
            "chop" : 4, #for Ry(pi/2), power Rabi 
            "rescaling" : 1., # Wigner of vacuum amplitude
        }
        wigner_params = prepare_wigner_operators(wigner_params_0,
                                                state_params, 
                                                cdim, 
                                                qdim)
        
        params = state_params | wigner_params
        print(params["T2"])

        beta_list = np.linspace(-2.85, 2.85, 41) 
        B1, B2 = np.meshgrid(beta_list, beta_list, indexing="ij")
        beta_pairs = np.column_stack([B1.ravel(), B2.ravel()])

        nproc = max(1, cpu_count() - 1)
        with Pool(processes=nproc, initializer=init_worker, initargs=(params,)) as pool:
            W_flat = pool.map(run_simulation, beta_pairs)

        W = np.array(W_flat, dtype=float).reshape(len(beta_list), len(beta_list))
        W /= wigner_params["rescaling"]
        
        # Save wigner to npz
        rho = state_params["rho"]
        chi = wigner_params["chi"]
        Kerr = wigner_params["Kerr"]
        alpha = wigner_params["alpha"]
        wigner_wait = wigner_params["wigner_wait"]
        sigma = wigner_params["sigma"]
        chop = wigner_params["chop"]
        rescaling = wigner_params["rescaling"]

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

        
        end = time.time()
        print(f"Execution time: {end - start:.3f} s")

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