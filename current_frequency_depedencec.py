import numpy as np
from sympy import Symbol, sympify, symbols, diff, Wild, expand
import sympy as sp
from sympy.utilities.lambdify import lambdify
from scipy.optimize import curve_fit
from scipy.signal import butter, filtfilt
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import csv
import os
from copy import deepcopy

def calculate_MSE(y, y_pred):
    """Returns the mean squared error between a y value (real) and a predicted y value (prediction)."""
    mse = sum((y_pred - y)**2)
    return mse

def calculate_MSE_complex(y, y_pred):
    """Returns the mean squared error between a complex y value and a predicted complex y value."""
    mse = sum(np.abs(y_pred - y)**2)
    return mse

def load_data(Hz):
    """
    Returns numpy arrays of time_data, voltage, current and frequency for a set of data files
    """
    loc=r".\Data_for_eq_learning"

    #loading data
    data_volt = pd.read_csv(f"Data_for_eq_learning/{Hz}_Hz_2_cv_voltage", sep="\t", names = ["time","voltage"])
    data_amp = pd.read_csv(f"Data_for_eq_learning/{Hz}_Hz_2_cv_current", sep="\t", names = ["time","current"])
    data_combined = data_volt.join(data_amp["current"])

    #slicing data to remove start and end noise. Putting data into tuples
    slice_start = 550
    slice_end = -100
    time_data = np.array(data_combined["time"].iloc[slice_start:slice_end])
    voltage = np.array(data_combined["voltage"].iloc[slice_start:slice_end])
    current = np.array(data_combined["current"].iloc[slice_start:slice_end])

    return time_data, voltage, current

def fourier_transform(current, time_data):
   
    ft=np.fft.fft(current)
    freqs=np.fft.fftfreq(len(time_data), time_data[1]-time_data[0])

    return freqs, ft
    
def filter_fft(freqs, ft, band_size, Hz, desired_harmonic):
    freq_loc=desired_harmonic*int(Hz)
    filtered_ft=np.zeros(len(ft), dtype="complex")
    for sign in [-1, 1]:
        ft_loc=np.where((freqs>sign*freq_loc-(band_size*int(Hz))) & (freqs<sign*freq_loc+(band_size*int(Hz))))
        filtered_ft[ft_loc]=ft[ft_loc]
    
    return filtered_ft

def fit_sin(tt, yy):
    """
    Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"
    Obtained from https://stackoverflow.com/questions/16716302/how-do-i-fit-a-sine-curve-to-my-data-with-pylab-and-numpy, user unsym, Feb 19, 2017

    """
    tt = np.array(tt)
    yy = np.array(yy)
    ff = np.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing
    Fyy = abs(np.fft.fft(yy))
    guess_freq = abs(ff[np.argmax(Fyy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
    guess_amp = np.std(yy) * 2.**0.5
    guess_offset = np.mean(yy)
    guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset])

    def sinfunc(t, A, w, p, c):  return A * np.sin(w*t + p) + c
    popt, pcov = curve_fit(sinfunc, tt, yy, p0=guess)
    A, w, p, c = popt
    f = w/(2.*np.pi)
    fitfunc = lambda t: A * np.sin(w*t + p) + c
    output_dict = {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1./f, "fitfunc": fitfunc, "maxcov": np.max(pcov), "rawres": (guess,popt,pcov)}

    #lambdifying
    x0 = Symbol("x0")
    voltage_equation = A*sp.sin(w*x0 + p) + c

    return voltage_equation, [A,w,p,c]

def find_amp_phase(data, time_data, w, fs):
    v = data
    w = w
    ref_sin = np.sin(w * time_data)
    ref_cos = np.cos(w * time_data)

    print(w/(2*np.pi))

    #butterworth low pass filter
    b, a = butter(N=4, Wn = w/(2*np.pi), fs = fs)
    X = filtfilt(b, a, v*ref_sin)
    Y = filtfilt(b, a, v*ref_cos)

    fig, axs = plt.subplots()
    axs.plot(time_data, v*ref_sin, label = "Ref sin signal")
    axs.plot(time_data, X, label = "Filtered")
    axs.set_xlabel("time")
    axs.set_ylabel("signal")
    axs.legend()
    fig.tight_layout()
    Path(os.path.join(output_filepath,str(Hz))).mkdir(parents=True, exist_ok=True)
    plt.savefig(os.path.join(output_filepath, str(Hz), f"{Hz}Hz-filter-sin.png"))
    plt.close(fig)

    fig, axs = plt.subplots()
    axs.plot(time_data, v*ref_cos, label = "Ref cos signal")
    axs.plot(time_data, Y, label = "Filtered")
    axs.set_xlabel("time")
    axs.set_ylabel("signal")
    axs.legend()
    fig.tight_layout()
    plt.savefig(os.path.join(output_filepath, str(Hz), f"{Hz}Hz-filter-cos.png"))
    plt.close(fig)

    X = np.median(X)
    Y = np.median(Y)
    A = 2*np.sqrt(X**2+Y**2)
    p = np.arctan(Y/X)

    return A, p

#####
files_freq = [9, 36, 45, 54, 63, 72, 81, 90, 99]
output_filepath = rf"results\20250617-current-phase-original-set"
Path(os.path.join(output_filepath)).mkdir(parents=True, exist_ok=True)

with open(os.path.join(output_filepath,"summary.csv"),"a") as file:
    writer = csv.writer(file)
    writer.writerow([
                    "Hz",
                    "frequency",
                    "amplitude",
                    "phase",
                    ])

frequency = []
phase = []
amplitude = []
fs = 100000
for Hz in files_freq:
    time_data, voltage, current = load_data(Hz)
    voltage_eqn, coeff_array = fit_sin(time_data, voltage)
    A, p = find_amp_phase(current, time_data, coeff_array[1], fs)

    frequency.append(coeff_array[1]/(2*np.pi))
    phase.append(p)
    amplitude.append(A)

    with open(os.path.join(output_filepath,"summary.csv"),"a") as file:
        writer = csv.writer(file)
        writer.writerow([
                        Hz,
                        coeff_array[1]/(2*np.pi),
                        A,
                        p,
                        ])

fig, axs = plt.subplots()
axs.plot(frequency, amplitude, label = "Amplitude")
axs.scatter(frequency, amplitude)
axs.set_xlabel("frequency")
axs.set_ylabel("amplitude")
axs.legend()
fig.tight_layout()
plt.savefig(os.path.join(output_filepath, f"9-99-current-amplitude-frequency-dependence.png"))

fig, axs = plt.subplots()
axs.plot(frequency, phase, label = "Phase")
axs.scatter(frequency, phase)
axs.set_xlabel("frequency")
axs.set_ylabel("phase")
axs.legend()
fig.tight_layout()
plt.savefig(os.path.join(output_filepath, f"9-99-current-phase-frequency-dependence.png"))








""" #loading data
data = pd.read_csv(r"Data_for_eq_learning\20250611-CjX\20250611-blank-multifreq.csv", header = 0)

#slicing data to remove start and end noise. Putting data into tuples
slice_start = 0
slice_end = -1
time_data_new = np.array(data["x"].iloc[slice_start:slice_end])
voltage_new = np.array(data["z"].iloc[slice_start:slice_end])
current_new = np.array(data["y"].iloc[slice_start:slice_end]) """


