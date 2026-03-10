import numpy as np
import scipy as sc
import qutip
erf = sc.special.erf

def calibrate_pi(sigma, chop, alpha, T1, Tphi, qdim):
    '''
    Power Rabi: Use this to calibrate the amplitude needed to drive a qubit pi pulse
    '''
    ug = qutip.basis(qdim, 0)
    ue = qutip.basis(qdim, 1)
    q = qutip.destroy(qdim)
    qd = q.dag()

    amp = np.linspace(0, 1.5, 199)
    output = []

    
    def pulse(t, *arg):
        t0 = sigma*chop/2

        g = np.exp( - 1/2 * (t - t0)**2 / sigma**2)

        return g

    tlist = np.linspace(0, sigma*chop, 100) # in ns
    
    for Ax in amp:
        A = np.sqrt(2/np.pi) / erf(np.sqrt(2))*np.pi/(4*sigma)/2/np.pi#initial guess
        A0 = A#keep it for later

        freq = 0#resonant driving

        A *= Ax#coefficient for the Gaussian pulse

        H0 = 2*np.pi*freq * qd*q - 2*np.pi*alpha/2 * qd*qd*q*q
        Hd = 2*np.pi*A*1j*(qd - q)#or with other part Hd = 2*np.pi*A*(qd + q)


        H = [H0, [Hd, pulse]]

        psi = qutip.basis(3, 0)#initial state
        rhoq = qutip.ket2dm(psi)

        c_ops = [
            np.sqrt(1/T1)*q,
            np.sqrt(2/Tphi)*qd*q # changed
        ]
        e_ops = [ue*ue.dag(),]

        # options = Options(max_step = 1, nsteps = 1e6)

        results = qutip.mesolve(H, rhoq, tlist, c_ops = c_ops, e_ops = e_ops)#, options= options)#, progress_bar = True)

        output += [results.expect[0][-1],]
    
    A = A0*amp[output.index(max(output))]#this is the correct coeff
    return A, pulse