import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
from qutip import *

def gaussian(x, A, sigma, offset, x_offset):
    return A * np.exp(-((x + x_offset) ** 2) / 2 / sigma**2) + offset

def read_hdf5_file(file_path, data):
    with h5py.File(file_path, "r") as hdf:
        data_single_shot = hdf[data]
        cavity_drive_I = hdf["x_disp"]
        cavity_drive_Q = hdf["y_disp"]
        return (
            data_single_shot[ :, :, :],
            cavity_drive_I[:],
            cavity_drive_Q[:],
        )

data_path = "Valadares-Dorogov-FAR/Sfig_7_qudit_arbitrary_state_preparation/data/"
def save_state_pdf(filename, out_filename, correction, data = "Q"):

    (
        data_single_shot_all,
        cavity_drive_I,
        cavity_drive_Q,
    ) = read_hdf5_file(data_path + filename, data)

    amp, sigma, ofs = correction

    parity_corrected = (
        data_single_shot_all[:, :, 0] - data_single_shot_all[:, :, 1] - ofs
    ) / amp
    displacement_scale = 1.5

    fig = plt.figure()
    plt.xlabel("I displacement")
    plt.ylabel("Q displacement")
    cf = plt.pcolormesh(
        cavity_drive_I * displacement_scale,
        cavity_drive_Q * displacement_scale,
        parity_corrected,
        cmap="bwr",
        vmax=1,
        vmin=-1,
    )
    plt.title(f"Max = {np.max(parity_corrected):.3f}, min = {np.min(parity_corrected):.3f}")
    plt.gca().set_aspect("equal")
    plt.savefig(out_filename, bbox_inches="tight", transparent=True)
    plt.show()


correction_1 = (7.566813683222962e-05,  0.34714904442371364, -3.637468551906393e-07)
correction_2 = (0.00012211092624652933*0.530/0.719,  0.3771305102174895, -3.37138383378075e-07)
correction_3 = (0.0001356224263368787,  0.37349886800654697, 2.847278113888571e-07)
correction_4 = (0.00012366574221727816,  0.37137971245469314, -1.6081591296779986e-07)
correction_5 = (0.00012828291689372586,  0.3926357911725215, 3.294260470727042e-07)

path = "Valadares-Dorogov-FAR/Sfig_7_qudit_arbitrary_state_preparation/Wigner_tomographies/"

filename = "20250703_071105_lakeside_3.h5"
save_state_pdf(filename, path + "fock_3.pdf", correction_1, data = "Q")

filename = "20250702_232609_lakeside_5.h5"
save_state_pdf(filename, path + "fock_5.pdf", correction_1, data = "Q")

filename = "20250704_182756_lakeside_7.h5"
save_state_pdf(filename, path + "fock_7.pdf", correction_1, data = "Q")

filename = "20250703_013356_lakeside_0p1.h5"
save_state_pdf(filename, path + "fock_0p1.pdf", correction_1, data = "Q")

filename = "20250703_163842_lakeside_0p3.h5"
save_state_pdf(filename, path + "fock_0p3.pdf", correction_3, data = "Q")

filename = "20250705_015400_lakeside_0p7.h5"
save_state_pdf(filename, path + "fock_0p7.pdf", correction_1, data = "Q")

filename = "20250703_203049_lakeside_3p5.h5"
save_state_pdf(filename, path + "fock_3p5.pdf", correction_4, data = "Q")

filename = "20250705_171343_lakeside_3p7.h5"
save_state_pdf(filename, path + "fock_3p7.pdf", correction_2, data = "Q")

filename = "20250706_001428_lakeside_1p7.h5"
save_state_pdf(filename, path + "fock_1p7.pdf", correction_2, data = "Q")

filename = "20250707_134117_lakeside_1p5.h5"
save_state_pdf(filename, path + "fock_1p5.pdf", correction_5, data = "Q")
