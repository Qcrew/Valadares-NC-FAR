import numpy as np

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
