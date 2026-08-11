import numpy as np
from scipy.ndimage import gaussian_filter1d
import lmfit
from lmfit import Model
import h5py
import pylab as plt

def read_hdf5_file(file_path, x_axis_name: str = "idling_time",
                   y_axis_name: str = "flux_wait_ampx"):
    with h5py.File(file_path, "r") as hdf:
        # print("Keys: %s" % hdf.keys())
        x_ = hdf[x_axis_name]
        y_ = hdf[y_axis_name]
        z_ = hdf["I"]
        return (
            np.array(z_[:, :]),
            x_[:],
            y_[:],
        )

def sine(y, x, return_params=False):
    """ """

    def fn(x, f0, ofs, amp, phi):
        """ """
        return ofs + amp * np.sin(2 * np.pi * f0 * x + phi)

    def params(y, x):
        """ """
        fs = np.fft.rfftfreq(len(x), x[1] - x[0])
        ofs = np.mean(y)
        fft = np.fft.rfft(y - ofs)
        idx = np.argmax(abs(fft))
        return create_params(
            f0={"value": fs[idx], "min": fs[0], "max": fs[-1]},
            ofs={"value": ofs, "min": np.min(y), "max": np.max(y)},
            amp={"value": np.std(y - ofs), "min": 0, "max": np.max(y) - np.min(y)},
            phi={"value": np.angle(fft[idx]), "min": -2 * np.pi, "max": 2 * np.pi},
        )

    fit_params = params(y, x)
    if return_params:
        return fit_params
    result = Model(fn).fit(y, fit_params, x=x)
    return result.best_fit, result.best_values

def exp_decay_sine(x, amp=1, f0=0.05, phi=np.pi / 4, ofs=0, tau=0.5):
        return amp * np.sin(2 * np.pi * x * f0 + phi) * np.exp(-x / tau) + ofs

def exp_decay_sine_fit(y, x, fit_function=exp_decay_sine):
    """ """
    def params(y, x):
        """ """
        params = sine(y, x, return_params=True)
        params.add("tau", value=np.average(x), min=0, max=10 * x[-1])
        return params

    result = Model(fit_function).fit(y, params(y, x), x=x)
    return result.best_fit, result.best_values

def create_params(**kwargs):
    """patch method because lmfit does not like working with np datatypes"""
    params = {}
    for name, value in kwargs.items():
        if isinstance(value, dict):
            value = {k: v.item() for k, v in value.items() if isinstance(v, np.number)}
        elif isinstance(value, np.number):
            value = value.item()
        params[name] = value
    return lmfit.create_params(**params)

signal_I, idling_time, flux_ampx = read_hdf5_file("23-11-52_T2_Cavity_Resonant_long_averaged.h5")
idling_time_us = idling_time*4e-3
for ii, flux_amp in enumerate(flux_ampx):
    if flux_amp == 0:
        signal_I_e6 = signal_I[ii, :]*1e6
        best_fit, best_values = exp_decay_sine_fit(signal_I_e6, idling_time_us)
        smoothened_times = np.linspace(start=idling_time_us[0], stop=idling_time_us[-1], num=40_000)
        CM = 1/2.54
        fig = plt.figure(figsize=(17.2*CM, 5*CM))
        ax = plt.gca()
        ax.plot(idling_time_us, signal_I_e6, color="#88b742", marker="o", markersize=2, linewidth=0.5,
                    label="Data")
        ax.plot(smoothened_times, exp_decay_sine(smoothened_times, **best_values),
                color="#bc343a", linestyle="--", linewidth=1.0,
                label=r"Fit: $f_0$: {:.5f} MHz, ofs: {:.1f}, $T_2$: {:.5f} $\mu$s".format(best_values['f0'], best_values['ofs'], best_values['tau']))
        # ax.legend()
        ax.tick_params(axis="x", labelbottom=False, bottom=True, top=True)
        ax.tick_params(axis="y", labelleft=False, left=True, right=True)
        ax.set(xticks=[0, 10, 20, 30, 40], yticks=[40, 45, 50], xlim=[-1, 41])
        plt.tight_layout()
        plt.show()
        # fig.savefig("T2_legend.pdf",
        #                     bbox_inches="tight", transparent=True)