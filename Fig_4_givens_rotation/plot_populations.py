#%% The code to reproduce the population plots, Fig. 4 (c)
import numpy as np
import pylab as plt
import qutip
import pickle

def get_populations(input_density_matrix: qutip.Qobj, projectors: list = None) -> list:
    """
    Retrieve the Fock states populations from the density matrices
    Arguments:
    _____________________________
    input_density_matrix: qutip.Qobj -- the density matrix of the state to infer the populations
    projectors: list -- the list of projectors on the states, whose populations are of interest. If None, projects to the first 6 Fock states. Defaults to None

    Returns:
    _____________________________
    population_vector: list -- the list of populations
    """
    if projectors is None:
        loc_projectors = []
        for ii in range(6):
            loc_projectors.append(qutip.basis(7, ii).proj())
    else:
        loc_projectors = projectors
    population_vector = []
    for projector in loc_projectors:
        population_vector.append((input_density_matrix*projector).tr())
    return population_vector

def get_pops_from_simulation(filename_: str) -> list:
    """
    Retrieve the simulated density matrices from the file
    Arguments:
    _____________________________
    filename_: str -- the name of the file containing the simulated density matrix
    
    Returns:
    _____________________________
    read_populations: list -- the list of populations
    """
    read_file = np.load(filename_, allow_pickle=True)
    read_rho = qutip.Qobj(read_file['rho'])
    read_populations = get_populations(input_density_matrix=read_rho, projectors=projectors)
    return read_populations

# Load the reconstructed density matrices
with open("saved_density_matrices/1_plus_4_1.pickle", 'rb') as file:
    step_1 = pickle.load(file)

with open("saved_density_matrices/1_plus_4_2.pickle", 'rb') as file:
    step_2 = pickle.load(file)

with open("saved_density_matrices/1_plus_4_3.pickle", 'rb') as file:
    step_3 = pickle.load(file)

with open("saved_density_matrices/1_plus_4_4.pickle", 'rb') as file:
    step_4 = pickle.load(file)
states_vector = [step_1, step_2, step_3, step_4]

# Hilbert space truncation
D = 7

# Defining the projectors
projectors = []
for ii in range(D-1):
    projectors.append(qutip.basis(D, ii).proj())

# Defining the perfect states 
perfect_state_1 = qutip.ket2dm(qutip.basis(D, 1))
perfect_state_2 = qutip.ket2dm((qutip.basis(D,1) * np.cos(np.pi/8) + qutip.basis(D,4) * np.sin(np.pi/8)).unit())
perfect_state_3 = qutip.ket2dm((qutip.basis(D,1) + qutip.basis(D,4)).unit())
perfect_state_4 = qutip.ket2dm(qutip.basis(D, 4))
perfect_states_vector = [perfect_state_1, perfect_state_2, perfect_state_3, perfect_state_4]

# Retrieving the experimental populations
populations = []
for state_ in states_vector:
    populations.append(get_populations(input_density_matrix=state_, projectors=projectors))

# Retrieving the target states populations
perfect_populations = []
for state_ in perfect_states_vector:
    perfect_populations.append(get_populations(input_density_matrix=state_, projectors=projectors))

# Retrieving the simulated states populations
filenames_sim = ["reconstructed_1to4_theta0_T22700_T115000_cavT1_500000.npz", "reconstructed_1to4_thetapi4_T22700_T115000_cavT1_500000.npz",
                      "reconstructed_1to4_thetapi2_T22700_T115000_cavT1_500000.npz", "reconstructed_1to4_thetapi_T22700_T115000_cavT1_500000.npz"]
foldername = "simulated_density_matrices/"
simulated_populations = []
for _filename in filenames_sim:
    simulated_populations.append(get_pops_from_simulation(filename_=foldername+_filename))

# Preparation for plotting
indices = np.arange(len(projectors))
CM = 1 / 2.54
colours = np.array(["#bc343a", "#f19f3c", "#88b742", "#3E9491", "#7c1954"])

#%% Plotting the stacked bar charts of the populations
# fig = plt.figure(figsize=(2.435*CM, 2.435*CM))
fig = plt.figure(figsize=(5*CM, 5*CM))
ax = plt.gca()
plot_index = 3
target_indices = [1, 4]
non_target_indices = [0, 2, 3]
tiny_colours_list = np.array([colours[2], colours[4], colours[1]])
for ii in indices:
    perfect_pop = perfect_populations[plot_index][ii]
    sim_pop = simulated_populations[plot_index][ii]
    exp_pop = populations[plot_index][ii]
    tiny_pop_list = np.array([exp_pop, sim_pop, perfect_pop])
    for jj in np.argsort(tiny_pop_list)[::-1]:
        ax.bar(ii, tiny_pop_list[jj], color=tiny_colours_list[jj], edgecolor=None, alpha=1.0)
ax.set(ylim=[0, 1.05], xticks=[2], yticks=[0.5])
ax.tick_params(axis='x', labelbottom=False)
ax.tick_params(axis='y', labelleft=False)
# ax.legend()
fig.savefig("Figures/Stacked_bar_charts/Stacked_state_{}.pdf".format(plot_index+1), bbox_inches="tight", transparent=True)