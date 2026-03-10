import numpy as np
from tools.hamiltonian import E_list


def build_opt_drive_envelope(transition_freqs_list, drive_amps_list, M, t_max, t_off, phase_off):

    num_freqs = len(transition_freqs_list)
    wnq = 2*np.pi*transition_freqs_list
    
    # drive_ams_list is given in MHz. Here it is converted to 2pi*GHz
    Man = drive_amps_list[:num_freqs*M].reshape([num_freqs, M]) * 2 * np.pi * 1e-3
    Mbn = drive_amps_list[num_freqs*M:].reshape([num_freqs, M]) * 2 * np.pi * 1e-3

    def An(t, an):
        base = 2 * np.pi * t / t_max
        return sum(an[k] * (1 - np.cos((k + 1) * base)) for k in range(M))

    def Bn(t, bn):
        base = 2 * np.pi * t / t_max
        return sum(bn[k] * np.sin((k + 1) * base) for k in range(M))
    
    def build_coeffs(wnq, Man, Mbn):
        def H1_coeff(t, args):

            return sum(0.5 * (An(t, Man[n]) + 1j * Bn(t, Mbn[n])) * np.exp(-1j *( wnq[n] * (t + t_off) + 2 * np.pi * phase_off))
                       for n in range(num_freqs))
            # return sum(0.5 * (An(t, Man[n]) + 1j * Bn(t, Mbn[n])) * np.exp(-1j * wnq[n] * t) for n in range(num_freqs))
        
        def H2_coeff(t, args):
            return sum(0.5 * (An(t, Man[n]) - 1j * Bn(t, Mbn[n])) * np.exp(1j * ( wnq[n] * (t + t_off) + 2 * np.pi * phase_off)) 
                       for n in range(num_freqs))
            # return sum(0.5 * (An(t, Man[n]) - 1j * Bn(t, Mbn[n])) * np.exp(1j * wnq[n] * t) for n in range(num_freqs))
        
        return H1_coeff, H2_coeff

    H_coeff_1, H_coeff_2 = build_coeffs(wnq, Man, Mbn)
    return H_coeff_1, H_coeff_2


