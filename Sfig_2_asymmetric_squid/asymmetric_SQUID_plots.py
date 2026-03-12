#%% Imports
import numpy as np
import pylab as plt
import qutip
import time
from Aleksandr_SQUID_transmon_simulation_routines import *
#%% gamma = 1
start_time = time.time()
gamma = 1
N_freq_to_plot = 1
phis_e = np.linspace(-1, 1, 600)
Lj = 13e-9
C = 92e-15
eigenenergies = np.array([asym_H(gamma=gamma, phi_e=ext_flux, Lj=Lj, C=C, N=300).eigenenergies() for ext_flux in phis_e])
print("\n--- %s seconds ---" % (time.time() - start_time))
#%% gamma = 3
start_time = time.time()
L_1 = 18e-9
L_2 = 6e-9
C = 200e-15
gamma = L_1 / L_2
eigenenergies_gamma = np.array([asym_H(gamma=gamma, phi_e=ext_flux, Lj=L_1, C=C, N=300).eigenenergies() for ext_flux in phis_e])
print("\n--- %s seconds ---" % (time.time() - start_time))
#%% gamma = 6
start_time = time.time()
L_1 = 48e-9
L_2 = 8e-9
C = 200e-15
gamma = L_1 / L_2
eigenenergies_gamma_6 = np.array([asym_H(gamma=gamma, phi_e=ext_flux, Lj=L_1, C=C, N=300).eigenenergies() for ext_flux in phis_e])
print("\n--- %s seconds ---" % (time.time() - start_time))
#%% Plotting the spectra
CM = 1 / 2.54
FONTSIZE = 11
fig = plt.figure(figsize = (10.5*CM, 6.9*CM))
ax = plt.gca()
for ii, vals in enumerate(np.diff(eigenenergies).T[:N_freq_to_plot]):
    ax.plot(phis_e, vals/1e9,
            label=r"$\gamma = 1$",
            color=[0.094, 0.29, 0.27])
for ii, vals in enumerate(np.diff(eigenenergies_gamma).T[:N_freq_to_plot]):
    ax.plot(phis_e, vals/1e9,
            label=r"$\gamma = 3$",
            color=[0.25, 0, 1])
for ii, vals in enumerate(np.diff(eigenenergies_gamma_6).T[:N_freq_to_plot]):
    ax.plot(phis_e, vals/1e9,
            label=r"$\gamma = 6$",
            color=[0.71, 0.44, 0.44])

plt.xlabel(r"$\frac{\Phi_{ext}}{\Phi_0}$", fontsize=FONTSIZE)
plt.ylabel(r"$\omega$, GHz", fontsize=FONTSIZE)
ax.set_xticks(ticks=[-1, -0.5, 0, 0.5, 1])
ax.set_yticks(ticks=np.array([0, 2, 4, 6]))
ax.xaxis.label.set_fontsize(FONTSIZE)
ax.xaxis.set_tick_params(labelsize=FONTSIZE)
ax.yaxis.set_tick_params(labelsize=FONTSIZE)
ax.yaxis.label.set_fontsize(FONTSIZE)
ax.legend()
# fig.savefig('Asymmetric_SQUID_spectrum.pdf', bbox_inches='tight', transparent='True')
#%% Computing the flux sensitivities
start_time = time.time()
phis_e = np.linspace(-1, 1, 300)

gamma = 1
Lj = 13e-9
C = 92e-15
E_c, E_j = calculate_energies(Lj=Lj, C=C)
coefs_1 = np.array([flux_noise_coef(gamma=gamma, phi_e=phi_e, E_c=E_c, E_j=E_j) for phi_e in phis_e]) / 1e25

gamma = 3
Lj = 18e-9
C = 200e-15
ylim=[-5, 5]
E_c, E_j = calculate_energies(Lj=Lj, C=C)
coefs_3 = np.array([flux_noise_coef(gamma=gamma, phi_e=phi_e, E_c=E_c, E_j=E_j) for phi_e in phis_e]) / 1e25

gamma = 6
Lj = 48e-9
E_c, E_j = calculate_energies(Lj=Lj, C=C)
coefs_6 = np.array([flux_noise_coef(gamma=gamma, phi_e=phi_e, E_c=E_c, E_j=E_j) for phi_e in phis_e]) / 1e25
print("\n--- %s seconds ---" % (time.time() - start_time))
#%% Plotting the flux sensitivities
CM = 1 / 2.54
FONTSIZE = 11
fig = plt.figure(figsize=(10.5*CM, 6.9*CM))
ax = plt.gca()
ax.plot(phis_e, coefs_1, color=[0.094, 0.29, 0.27], label=r"$\gamma=1$")
ax.plot(phis_e, coefs_3, color=[0.25, 0, 1], label=r"$\gamma=3$")
ax.plot(phis_e, coefs_6, color=[0.71, 0.44, 0.44], label=r"$\gamma=6$")
plt.ylim(ylim)
ax.set(xlabel=r"$\frac{\Phi_{ext}}{\Phi_0}$",
       ylabel=r"$\frac{\partial \omega_{01}}{\partial \Phi_e}$, $10^{25}$ SI units")
ax.legend()
ax.set_xticks(ticks=[-1, -0.5, 0, 0.5, 1])
ax.xaxis.label.set_fontsize(FONTSIZE)
ax.xaxis.set_tick_params(labelsize=FONTSIZE)
ax.yaxis.set_tick_params(labelsize=FONTSIZE)
ax.yaxis.label.set_fontsize(FONTSIZE)
ax.title.set_fontsize(FONTSIZE)
# fig.savefig('Flux_noise_coefs.pdf', bbox_inches='tight', transparent='True')