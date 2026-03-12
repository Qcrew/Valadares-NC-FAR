#%%
import numpy as np
import pylab as plt
import qutip
from Aleksandr_SQUID_transmon_simulation_routines import *
gamma = 4.02
N_freq_to_plot = 2
# filename='Transmon_spectrum_plot.pdf'
filename = None
phis_e = np.linspace(-1, 1, 300)
plot_asym_freq(gamma=gamma, spectrum_param=phis_e, Lj=13.66e-9, C=154.7e-15, N_freq_to_plot=N_freq_to_plot,
                plot_analytical=True, filename=filename)