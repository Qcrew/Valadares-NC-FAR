"""
© Aleksandr Dorogov, 2026
"""
import numpy as np
import pylab as plt
import qutip
import scipy.constants
def _Ec(C:float = 66.22e-15) -> float:
    """Calculates capacitive energy in units of frequency

    Args:
        C (float): capacitance [F]. Defaults to 66.22 fF

    Returns:
        float: capacitive energy e^2 / (2 * C) / h
    """
    return scipy.constants.e**2 / (2 * C * scipy.constants.h)

def _Ej(Lj: float = 8.2e-9) -> float:
    """Calculates Josephson energy in units of frequency

    Args:
        Lj (float): inductance. Defaults to 8.2 nH

    Returns:
        float: Josephson energy (Phi_0/2pi)^2 / Lj / h
    """
    return (scipy.constants.physical_constants['mag. flux quantum'][0] / 2 / scipy.constants.pi)**2 / (Lj * scipy.constants.h)
def ChargeQubitNg(ng, Ec, Ej, N = 100):
    """Returns charge qubit Hamiltonian with ng
    
    Args:
        ng (float): qubit parameter
        Ec (float): qubit parameter
        Ej (float): qubit parameter
        N (int): defines Hamiltonian shape as (2N+1, 2N+1)
        
    Returns:
        qobj: charge qubit hamiltonian
             with ng in the charge basis
             
    """
    return 4 * Ec * (qutip.charge(N) - ng)**2 - 0.5 * Ej * qutip.tunneling(2 * N + 1)

def eff_Ej(Ej_1: float, gamma: float, phi_e: float) -> float:
    """Returns effective Josephson energy of asymmetric transmon
    Ej_1 (float): Josephson energy of smaller junction.
    gamma (float): energy ratio between two junctions.
    phi_e (float): bias magnetic flux in units of Phi_ext / Phi_0.
    """
    d = (gamma-1)/(gamma+1)
    phi_ext = scipy.constants.pi * phi_e
    return (gamma+1) * Ej_1 * np.sqrt(np.cos(phi_ext)**2 + d**2 * np.sin(phi_ext)**2)

def asym_H(gamma: float, phi_e: float, Lj: float = 8.2e-9, C: float = 66.22e-15,
           Ec: float = None, Ej: float = None, N: int = 40, ng: float = 0,
           print_E: bool = False) -> qutip.Qobj:
    """Returns asymmetric transmon Hamiltonian in charge basis.
    
    Args:
        gamma (float): energy ratio between two junctions.
        phi_e (float): bias magnetic flux in units of Phi_ext / Phi_0.
        Lj (float): inductance of the smaller Josephson junction. Defaults to 8.2 nH.
        C (float): capacitance of the qubit. Defaults to 66.22 fF.
        Ec (float): Capacitive energy of the transmon. If not None, overrides the expression through C. Defaults to None.
        Ej (float): Josephson energy of smaller junction. If not None, overrides the expression through Lj. Defaults to None.
        N (int): defines Hamiltonian shape as (2N+1, 2N+1). Defaults to 40.
        ng (float): qubit parameter. Defaults to 0.
        print_E (bool): whether to print E_c and E_j. Defaults to False.
        
    Returns:
        qobj: asymmetric transmon Hamiltonian with ng in the charge basis
             
    """
    if Ec is None:
        E_c = _Ec(C)
    else:
        E_c = Ec
    if Ej is None:
        E_j = _Ej(Lj)
    else:
        E_j = Ej
    eff_Josephson_energy = eff_Ej(E_j, gamma, phi_e)
    if print_E:
        print("E_J = {J_en:.2f} GHz".format(J_en=eff_Josephson_energy/1e9))
        print("E_C = {C_en:.2f} GHz".format(C_en=E_c/1e9))
        print("E_J / E_C = {ratio:.1f}".format(ratio=eff_Josephson_energy/E_c))
    return 4 * E_c * (qutip.charge(N) - ng)**2 - 0.5 * eff_Josephson_energy * qutip.tunneling(2 * N + 1)

def plot_transition_energies(vals, nfirst = 10):
    """Plots the energy difference as a function of energy
    
    Args:
        vals (array): list of energy eigenvalues
        nfirst (int): number of energy levels to plot

    """
    fig, axes = plt.subplots(1, 2, figsize = (10, 8))
    for ii in range(nfirst):
        axes[0].plot(np.linspace(-5, 5, 21), vals[ii] * np.ones(21), label=f'Level {ii}')
    energy_jumps = np.diff(vals)
    axes[1].plot(range(nfirst), energy_jumps[:nfirst], color='darkblue', linestyle='dotted', marker='x')
    axes[0].tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
    axes[1].minorticks_on()
    axes[1].grid(which='both')
    axes[0].set(ylabel='Eigenenergy')
    axes[0].legend()
    axes[1].set(ylabel=r'Energy difference $E_{i+1} - E_i$', xlabel='i')

def plot_asym_freq(gamma: int, spectrum_param: np.ndarray, Lj: float = 8.2e-9, C: float = 66.22e-15,
                   Ec: float = None, Ej: float = None,
                   N_freq_to_plot: int = 5, xlabel: str = r"$\frac{\Phi_{ext}}{\Phi_0}$",
                   colormap: list = ['darkblue', 'red', 'black', 'green', 'purple', 'darkcyan', 'orange', 'brown', 'magenta', 'darkred', 'rebeccapurple', 'khaki', 'lime', 'cyan', 'pink'],
                   plot_analytical: bool =True,
                   filename: str = None):
    """Plots the energy spectrum
    
    Args:
        gamma (float): energy ratio between two junctions.
        spectrum_param (numpy.ndarray): external flux
        Lj (float): inductance of the smaller Josephson junction. Defaults to 8.2 nH.
        C (float): capacitance of the qubit. Defaults to 66.22 fF.
        Ec (float): Capacitive energy of the transmon. If not None, overrides the expression through C. Defaults to None.
        Ej (float): Josephson energy of smaller junction. If not None, overrides the expression through Lj. Defaults to None.
        N_freq_to_plot (int): number of energy levels to plot. Defaults to 5.
        xlabel (str): label of the x-axis. Defaults to r"$\phi_e$".
        colormap (list): a list of colors for plotting. Defaults to ['darkblue', 'red', 'black', 'green', 'purple', 'darkcyan', 'orange', 'brown', 'magenta', 'lime', 'darkred', 'rebeccapurple', 'khaki', 'cyan', 'pink'].
        plot_analytical (bool): whether to plot analytical expression for transition frequencies. Defaults to True.
        filename (str): name of the file to save to. If None, the plot won't be saved. Defaults to None.
    """
    eigenenergies = np.array([asym_H(gamma=gamma, phi_e=ext_flux, Lj=Lj, C=C).eigenenergies() for ext_flux in spectrum_param])
    E_c, E_j = calculate_energies(Lj=Lj, C=C, Ec=Ec, Ej=Ej)
    highest_freq_GHz = analytical_upper_bound(gamma, E_c, E_j)/1e9
    lowest_freq_GHz = analytical_lower_bound(gamma, E_c, E_j)/1e9
    fig = plt.figure(figsize = (10, 8))
    ax = plt.gca()
    if plot_analytical:
        analytical_freqs = np.diff(np.array([[analytical_freq(m=energy_level, gamma=gamma,
                                                              phi_e=ext_flux, E_c=E_c, E_j=E_j) for energy_level in range(N_freq_to_plot+1)] for ext_flux in spectrum_param]))
        ax.plot(spectrum_param, highest_freq_GHz*np.ones(len(spectrum_param)),
                    color='lightsalmon', linestyle='dotted')
        ax.plot(spectrum_param, lowest_freq_GHz*np.ones(len(spectrum_param)),
                color='lightsalmon', linestyle='dotted')
    for ii, vals in enumerate(np.diff(eigenenergies).T[:N_freq_to_plot]):
        ax.plot(spectrum_param, vals/1e9,
                label=r"$\omega_{{{ground}{exc}}}^{{num}}$".format(ground=ii, exc=ii+1),
                color=colormap[ii])
        if plot_analytical:
            ax.plot(spectrum_param, analytical_freqs[:, ii]/1e9,
                    label=r"$\omega_{{{ground}{exc}}}^{{analyt}}$".format(ground=ii, exc=ii+1),
                    color=colormap[-ii-1], linestyle='dashed')
    ax.minorticks_on()
    ax.grid(which='both')
    ax.legend()
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(r"$\omega$, GHz", fontsize=12)
    plt.title(r"$\gamma = {asym_param:.2f}$".format(asym_param=gamma), fontsize=12)
    print('Frequency bounds are\n  {:.2f} GHz\n  {:.2f} GHz'.format(lowest_freq_GHz, highest_freq_GHz))
    if filename is not None:
        fig.savefig(filename, bbox_inches='tight')

def PlotNAverage(ng, eigenstates, nfirst = 3):
    """Plot the average charge as function of ng
    
    Args:
        ng (float): qubit parameter
        eigenstates (array): qubit eigenstates
        nfirst (int): number of energy levels to plot

    """
    fig = plt.figure(figsize = (10, 8))
    ax = plt.gca()
    eigen_charge = np.array([qutip.expect(qutip.charge((eigenstates.shape[2]-1)//2), specific_eigenstate[1][:nfirst]) for specific_eigenstate in eigenstates])
    print(eigen_charge.shape)
    for ii in range(len(eigen_charge[0, :])):
        ax.plot(ng, eigen_charge[:, ii], label=f'Level {ii}')
    ax.minorticks_on()
    ax.grid(which='both')
    ax.legend()
    plt.xlabel(r"$n_g$", fontsize=14)
    plt.ylabel(r"Average charge, $\langle n \rangle$", fontsize=14)

def analytical_freq(m: int, gamma: float, phi_e: float, E_c: float, E_j: float,
           print_E: bool = False) -> float:
    """Returns analytical values for transmon qubit (without constant offset for all energies)
    
    Args:
        m (int): the number of the energy level
        gamma (float): energy ratio between two junctions.
        phi_e (float): bias magnetic flux in units of Phi_ext / Phi_0.
        E_c (float): Capacitive energy of the transmon [Hz].
        E_j (float): Josephson energy of smaller junction [Hz].
        print_E (bool): whether to print E_c and E_j. Defaults to False.
        
    Returns:
        float: analytical values for transmon qubit (without constant offset for all energies) [Hz]
             
    """
    eff_Josephson_energy = eff_Ej(E_j, gamma, phi_e)
    if print_E:
        print("E_J = {J_en:.2f} GHz".format(J_en=eff_Josephson_energy/1e9))
        print("E_C = {C_en:.2f} GHz".format(C_en=E_c/1e9))
        print("E_J / E_C = {ratio:.1f}".format(ratio=eff_Josephson_energy/E_c))
    return m * (np.sqrt(8 * E_c * eff_Josephson_energy) - (m+1) * E_c / 2)

def calculate_energies(Lj: float = 8.2e-9, C: float = 66.22e-15,
           Ec: float = None, Ej: float = None) -> np.ndarray:
    """Calculates capacitive and Josephson energies of the asymmetric transmon for given parameters.

    Args:
        Lj (float): inductance of the smaller Josephson junction. Defaults to 8.2 nH.
        C (float): capacitance of the qubit. Defaults to 66.22 fF.
        Ec (float): Capacitive energy of the transmon. If not None, overrides the expression through C. Defaults to None.
        Ej (float): Josephson energy of smaller junction. If not None, overrides the expression through Lj. Defaults to None.
    """
    if Ec is None:
        E_c = _Ec(C)
    else:
        E_c = Ec
    if Ej is None:
        E_j = _Ej(Lj)
    else:
        E_j = Ej
    return np.array([E_c, E_j])

def analytical_upper_bound(gamma:float, E_c: float, E_j: float) -> float:
    """Returns the analytical expression for the highest achievable frequency for asymmetric transmon

    Args:
        gamma (float): energy ratio between two junctions.
        E_c (float): Capacitive energy of the transmon.
        E_j (float): Josephson energy of smaller junction.
    """
    return np.sqrt(8*E_c*E_j*(gamma+1)) - E_c

def analytical_lower_bound(gamma:float, E_c: float, E_j: float) -> float:
    """Returns the analytical expression for the lowest achievable frequency for asymmetric transmon

    Args:
        gamma (float): energy ratio between two junctions.
        E_c (float): Capacitive energy of the transmon.
        E_j (float): Josephson energy of smaller junction.
    """
    return np.sqrt(8*E_c*E_j*(gamma-1)) - E_c

def get_gamma_for_bandwidth(bandwidth: float, E_c: float, E_j: float) -> float:
    """Returns the gamma (asymmetry parameter) for given bandwidth and capacitive and Josephson energies

    Args:
        bandwidth (float): the difference between the highest and the lowest frequencies [Hz].
        E_c (float): Capacitive energy of the transmon.
        E_j (float): Josephson energy of smaller junction.

    Returns:
        gamma (float): energy ratio between two junctions.
    """
    eta = bandwidth / np.sqrt(8 * E_c * E_j)
    return eta**(-2) + (eta**2) / 4

def get_bandwidth(gamma: float, E_c: float, E_j: float) -> float:
    """Returns the bandwidth for asymmetric transmon for given asymmetry parameter and energies

    Args:
        gamma (float): energy ratio between two junctions.
        E_c (float): Capacitive energy of the transmon.
        E_j (float): Josephson energy of smaller junction.

    Returns:
        bandwidth (float): the difference between the highest and the lowest frequencies [Hz]
    """
    return np.sqrt(8 * E_c * E_j) * (np.sqrt(gamma+1) - np.sqrt(gamma - 1))

def flux_noise_coef(gamma: float, phi_e: float, E_c: float, E_j: float) -> float:
    """Returns the d\omega / d\Phi (flux noise coefficient) for asymmetric transmon for given asymmetry parameter and energies

    Args:
        gamma (float): energy ratio between two junctions.
        phi_e (float): external flux
        E_c (float): Capacitive energy of the transmon.
        E_j (float): Josephson energy of smaller junction.

    Returns:
        flux_coef (float): d\omega / d\Phi (flux noise coefficient)
    """
    d = (gamma - 1) / (gamma + 1)
    phi_ext = scipy.constants.pi * phi_e
    return scipy.constants.pi**2 / scipy.constants.physical_constants['mag. flux quantum'][0] * np.sqrt(2 * E_c * E_j * (gamma + 1)) * (d**2 - 1) * np.sin(2 * phi_ext) / (((np.cos(phi_ext))**2 + (d * np.sin(phi_ext))**2)**0.75)

def plot_flux_noise_coef(gamma: float, spectrum_param: np.ndarray, Lj: float = 8.2e-9, C: float = 66.22e-15,
                   Ec: float = None, Ej: float = None, figure = None, color: str = 'darkblue',
                   ylim: list = None, label: str = None, filename: str = None):
    """Plots the d\omega / d\Phi (flux noise coefficient) for asymmetric transmon for given asymmetry parameter and energies

    Args:
        gamma (float): energy ratio between two junctions.
        spectrum_param (numpy.ndarray): external flux.
        Lj (float): inductance of the smaller Josephson junction. Defaults to 8.2 nH.
        C (float): capacitance of the qubit. Defaults to 66.22 fF.
        Ec (float): Capacitive energy of the transmon. If not None, overrides the expression through C. Defaults to None.
        Ej (float): Josephson energy of smaller junction. If not None, overrides the expression through Lj. Defaults to None.
        figure (matplotlib.figure.Figure): matplotlib figure for the plot. If None, creates a new one with figsize = (10, 8). Defaults to None.
        color (str): colour of the line to plot. Defaults to 'darkblue'
        ylim (list): limits for the y-axis. If None, autoscale used. Defaults to None.
        label (str): label for the plot. Defaults to None.
        filename (str): name of the file to save to. If None, the plot won't be saved. Defaults to None.
    """
    E_c, E_j = calculate_energies(Lj=Lj, C=C, Ec=Ec, Ej=Ej)
    coefs = np.array([flux_noise_coef(gamma=gamma, phi_e=phi_e, E_c=E_c, E_j=E_j) for phi_e in spectrum_param])
    coef_bounds = [np.min(coefs), np.max(coefs)]
    if figure is None:
        fig = plt.figure(figsize=(10, 8))
    else:
        fig = figure
    ax = plt.gca()
    ax.plot(spectrum_param, coefs, color=color, label=label)
    ax.plot([-0.5]*2, coef_bounds, color='green', linestyle='dashed')
    ax.plot([0]*2, coef_bounds, color='green', linestyle='dashed')
    ax.plot([0.5]*2, coef_bounds, color='green', linestyle='dashed')
    ax.plot([spectrum_param[0], spectrum_param[-1]], [0]*2, color='green', linestyle='dashed')
    ax.set(xlabel=r"$\frac{\Phi_{ext}}{\Phi_0}$", ylabel=r"$\frac{\partial \omega_{01}}{\partial \Phi_e}$")
    if ylim is not None:
        plt.ylim(ylim)
    ax.minorticks_on()
    ax.grid(which='both')
    if label is not None:
        ax.legend()
    if filename is not None:
        fig.savefig(filename, bbox_inches='tight')

def get_freq_01(gamma: float, phi_e: float, E_c: float, E_j: float,) -> float:
    """Returns analytical value for the frequency of transmon qubit |g> -> |e> transition [Hz]
    
    Args:
        gamma (float): energy ratio between two junctions.
        phi_e (float): bias magnetic flux in units of Phi_ext / Phi_0.
        E_c (float): Capacitive energy of the transmon [Hz].
        E_j (float): Josephson energy of smaller junction [Hz].
        
    Returns:
        float: analytical value for the frequency of transmon qubit |g> -> |e> transition [Hz]
             
    """
    return analytical_freq(m=1, gamma=gamma, phi_e=phi_e, E_c=E_c, E_j=E_j, print_E=False) - analytical_freq(m=0, gamma=gamma, phi_e=phi_e, E_c=E_c, E_j=E_j, print_E=False)

def get_inductance_from_freq(freq: float, E_c: float) -> float:
    """Returns the qubit inductance [H] for given |g> -> |e> frequency [Hz] and capacitive energy [Hz]

    Args:
        freq (float): qubit |g> -> |e> frequency [Hz].
        E_c (float): Capacitive energy of the qubit [Hz].
        
    Returns:
        float: qubit inductance [H]
    
    """
    return 2 * E_c * (scipy.constants.physical_constants['mag. flux quantum'][0] / (scipy.constants.pi * (freq + E_c)))**2 / scipy.constants.h