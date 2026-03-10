import numpy as np
import matplotlib.pyplot as plt
from qutip import *

T1_list = [10000, 15000]
T2_list = [1300, 1400, 1500, 1600, 1700, 1800, 
           1900, 2000, 2100, 2200, 2300, 2400, 
           2500, 2600, 2700, 2800]

######### a0 + b2 + b4 #########

fid_matrix = np.array([
    [0.80475818, 0.8106699,  0.82031832, 0.82627257, 0.83132265, 0.84145696,
     0.84485065, 0.85035818, 0.85568866, 0.85886914, 0.86318478, 0.86520992,
     0.869869,   0.85597139, 0.87517089, 0.87838145],
    [0.80058442, 0.81098306, 0.8203524,  0.82769528, 0.83771427, 0.8437694,
     0.84431205, 0.85219788, 0.85525463, 0.86151389, 0.8626506,  0.8678191,
     0.86975651, 0.87255278, 0.8772684,  0.87799429]
])

fid_T1_A = 0.9372070294803413 # T1 = 10000, T2 = 19999
fid_T1_B = 0.9440976143353845 # T1 = 15000, T2 = 29999
fid_leak = 0.9510204311261463 # T1 = T2 = Tphi = np.inf

plt.scatter(T2_list, fid_matrix[0, :], label = f"T1 = {T1_list[0]/1000}us")
plt.scatter(T2_list, fid_matrix[1, :], label = f"T1 = {T1_list[1]/1000}us")
plt.plot([min(T2_list), max(T2_list)], [fid_T1_A, fid_T1_A], linestyle = "--", label = "T1 = 10 us, Tphi = inf")
plt.plot([min(T2_list), max(T2_list)], [fid_T1_B, fid_T1_B], linestyle = "--", label = "T1 = 15 us, Tphi = inf")
plt.plot([min(T2_list), max(T2_list)], [fid_leak, fid_leak], linestyle = "--", color = "k", label = "T1 = Tphi = inf")

plt.title("a0 + b2 + b4")
plt.ylabel("Fidelity")
plt.xlabel("T2")
plt.legend()
plt.show()


######### 0 + 3 #########

fid_matrix = np.array([
    [0.87195193, 0.87995334, 0.88716308, 0.89534449, 0.8994982,  0.9071827,
     0.91061872, 0.91397165, 0.91948187, 0.92202369, 0.92487926, 0.92680819,
     0.93017722, 0.93158307, 0.9365469,  0.93862615],
    [0.87256436, 0.87980022, 0.88721243, 0.89705271, 0.90109471, 0.90673871,
     0.909886,   0.91251929, 0.91853738, 0.92174565, 0.92608847, 0.92705569,
     0.93141775, 0.93350576, 0.93230882, 0.93789512]
])

fid_T1_A = 0.9789227549526234 # T1 = 10000, T2 = 19999
fid_T1_B = 0.9836942120004271 # T1 = 15000, T2 = 29999
fid_leak = 0.9840 # T1 = T2 = Tphi = np.inf

plt.scatter(T2_list, fid_matrix[0, :], label = f"T1 = {T1_list[0]/1000}us")
plt.scatter(T2_list, fid_matrix[1, :], label = f"T1 = {T1_list[1]/1000}us")
plt.plot([min(T2_list), max(T2_list)], [fid_T1_A, fid_T1_A], linestyle = "--", label = "T1 = 10 us, Tphi = inf")
plt.plot([min(T2_list), max(T2_list)], [fid_T1_B, fid_T1_B], linestyle = "--", label = "T1 = 15 us, Tphi = inf")
plt.plot([min(T2_list), max(T2_list)], [fid_leak, fid_leak], linestyle = "--", color = "k", label = "T1 = Tphi = inf")

plt.title("0 + 3")
plt.ylabel("Fidelity")
plt.xlabel("T2")
plt.legend()
plt.show()