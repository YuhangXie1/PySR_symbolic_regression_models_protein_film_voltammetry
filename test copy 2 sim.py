import numpy as np
from sympy import Symbol, sympify, symbols, diff, Wild, expand
import sympy as sp
from sympy.utilities.lambdify import lambdify
from scipy.optimize import curve_fit
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
    #loading data
    data_volt = pd.read_csv(f"Data_for_eq_learning/{Hz}_Hz_2_cv_voltage", sep="\t", names = ["time","voltage"])
    data_amp = pd.read_csv(f"Data_for_eq_learning/{Hz}_Hz_2_cv_current", sep="\t", names = ["time","current"])

    #slicing data to remove start and end noise. Putting data into tuples
    slice_start = 550
    slice_end = -100
    time_data = np.array(data_amp["time"].iloc[slice_start:slice_end])
    voltage = np.array(data_volt["voltage"].iloc[slice_start:slice_end])
    current = np.array(data_amp["current"].iloc[slice_start:slice_end])

    return time_data, voltage, current, [slice_start, slice_end]


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

def fit_voltage_eqn(time_data, voltage):
    voltage_eqn, coeff_array = fit_sin(time_data, voltage)
    
    x0 = Symbol("x0")
    dv_dt_eqn = diff(voltage_eqn, x0)

    return voltage_eqn, dv_dt_eqn, coeff_array
####
#files_freq = [9, 36, 45, 54, 63, 72, 81, 90, 99]
files_freq = [36,45,54,63,72]

fig, axs = plt.subplots()
fig2, axs2 = plt.subplots()
for Hz in files_freq:
    time_data, voltage, current, slice_array = load_data(Hz)
    voltage_eqn, dv_dt_eqn, coeff_array = fit_voltage_eqn(time_data, voltage)

    A = coeff_array[0]
    w = coeff_array[1]
    p = coeff_array[2]
    c = coeff_array[3]

    #current phase and amplitude data
    phase_data = pd.read_csv(rf"results\20250617-current-phase-original-set\summary.csv")
    p_val = np.array(phase_data.loc[phase_data["Hz"] == Hz]["phase"])[0]
    A_val = np.array(phase_data.loc[phase_data["Hz"] == Hz]["amplitude"])[0]

    x0 = Symbol("x0")
    dv_dt_lambda = lambdify(x0,dv_dt_eqn)
    dv_dt = dv_dt_lambda(time_data)


    voltage = np.sin(w*time_data)
    dv_dt = np.cos(w*time_data)

    #current = 3 * p_val * voltage * dv_dt + 2 * p_val * dv_dt + p_val *4*voltage**2 + p_val* voltage + p_val
    
    #time_data = time_data - (p_val/w)
    
    current = A_val*np.sin(w * time_data)

    #normalisation
    current = current/A_val


    axs.plot(time_data, current, label = f"{Hz}Hz")
    axs2.plot(dv_dt, current, label = f"{Hz}Hz")


axs.set_xlabel("time")
axs.set_ylabel("current")
axs.set_xlim(0,1)
axs.set_title(f"Files:{files_freq}-time-current")
axs.legend()
fig.tight_layout()

axs2.set_xlabel("dv_dt")
axs2.set_ylabel("current")
axs2.set_title(f"Files:{files_freq}-dv_dt-current")
axs2.legend()
fig2.tight_layout()
plt.show()





""" for Hz in files_freq:
    
    #load data
    time_data, voltage, current, slice_array = load_data(Hz)
    voltage_eqn, dv_dt_eqn, coeff_array = fit_voltage_eqn(time_data, voltage)


    #current phase and amplitude data
    phase_data = pd.read_csv(rf"results\20250617-current-phase-original-set\summary.csv")
    p_val = np.array(phase_data.loc[phase_data["Hz"] == Hz]["phase"])[0]
    A_val = np.array(phase_data.loc[phase_data["Hz"] == Hz]["amplitude"])[0]

    x0 = Symbol("x0")
    dv_dt_lambda = lambdify(x0,dv_dt_eqn)
    dv_dt = dv_dt_lambda(time_data)

    #scaling current
    current = current/A_val

    #scaling time by phase
    time_data = time_data + (p_val/coeff_array[1])
    dv_dt = dv_dt_lambda(time_data)

    axs.plot(dv_dt, current, label = f"{Hz}Hz") """




""" A = np.full((1,len(time_data)), coeff_array[0])[0]
    w = np.full((1,len(time_data)), coeff_array[1])[0]
    p = np.full((1,len(time_data)), coeff_array[2])[0]
    c = np.full((1,len(time_data)), coeff_array[3])[0]
    p_val = np.full((1,len(time_data)), p_val)[0]
    A_val = np.full((1,len(time_data)), A_val)[0] """
""" #calculating predicted current
x, dx, f = symbols("x dx f")
current_pred_lambda = lambdify([x, dx, f], eqn)
current_pred = current_pred_lambda(voltage, dv_dt, freq) """

""" band_size=0.1
freqs, ft = fourier_transform(current, time_data)
#freqs_pred, ft_pred = fourier_transform(current_pred, time_data)

#ploting all harmonics
fig, axs = plt.subplots()
axs.plot(freqs, np.log10(ft**2), color = "blue", label="real")
#axs.plot(freqs_pred, np.log10(ft_pred**2), color = "red", label="pred")
axs.set_xlim(0,400)
axs.set_ylim(-4, 3)
axs.set_xlabel("Frequency")
axs.set_ylabel("Log10 ft^2")
axs.set_title(f"FTacV all harmonics")
axs.legend()
fig.tight_layout()
plt.savefig(os.path.join(output_filepath, f"FTacV-all-harmonic-freq.png"))

#plotting figures 1 by 1
Hz = 36
for harmonic in range(1,10):
    fig, axs = plt.subplots()

    ft_filtered = filter_fft(freqs, ft, band_size, Hz, harmonic)
    #ft_filtered_pred = filter_fft(freqs_pred, ft_pred, band_size, Hz, harmonic)
    
    log_ft_filtered = np.log10(ft_filtered**2)
    #log_ft_filtered_pred = np.log10(ft_filtered_pred**2)
    
    axs.plot(freqs, log_ft_filtered, color = "blue", label="real")
    #axs.plot(freqs_pred, log_ft_filtered_pred, color = "red", label="pred")
    axs.set_xlim(harmonic*int(Hz)-harmonic*int(Hz)*0.03, harmonic*int(Hz)+harmonic*int(Hz)*0.03)
    axs.legend()
    axs.set_xlabel("Frequency (Hz)")
    axs.set_ylabel("Log10 ft^2")
    axs.set_title(f"FTacV harmonic {harmonic} in freq domain")
    fig.tight_layout()
    Path(os.path.join(output_filepath,"freq_domain")).mkdir(parents=True, exist_ok=True)
    plt.savefig(os.path.join(output_filepath, "freq_domain", f"FTacV-harmonic-{harmonic}-freq.png"))
    plt.close(fig.figure)


    inverseft=np.fft.ifft(ft_filtered)
    #inverseft_pred=np.fft.ifft(ft_filtered_pred)

    fig, axs = plt.subplots()
    axs.plot(time_data, inverseft, color = "blue", label="real")
    #axs.plot(time_data, inverseft_pred, color = "red", label="pred")
    axs.legend()
    axs.set_xlabel("Time")
    axs.set_ylabel("Current")
    axs.set_title(f"File {Hz}Hz harmonic {harmonic} in time domain")
    fig.tight_layout()
    Path(os.path.join(output_filepath,"time_domain")).mkdir(parents=True, exist_ok=True)
    plt.savefig(os.path.join(output_filepath,"time_domain", f"FTacV-harmonic-{harmonic}-time.png"))
    plt.close(fig.figure)


    fig, axs = plt.subplots()
    axs.plot(voltage, inverseft, color = "blue", label="real")
    #axs.plot(voltage, inverseft_pred, color = "red", label="pred")
    axs.legend()
    axs.set_xlabel("Voltage")
    axs.set_ylabel("Current")
    axs.set_title(f"File {Hz}Hz harmonic {harmonic} in voltage domain")
    fig.tight_layout()
    Path(os.path.join(output_filepath,"voltage_domain")).mkdir(parents=True, exist_ok=True)
    plt.savefig(os.path.join(output_filepath,"voltage_domain", f"FTacV-harmonic-{harmonic}-freq.png"))
    plt.close(fig.figure)

#Plotting 3 by 3 figures

fig1, axs1 = plt.subplots(3,3)
axs1 = axs1.flatten()

fig2, axs2 = plt.subplots(3,3)
axs2 = axs2.flatten()

fig3, axs3 = plt.subplots(3,3)
axs3 = axs3.flatten()

for harmonic in range(1,10):
    ax1 = axs1[harmonic - 1]

    ft_filtered = filter_fft(freqs, ft, band_size, Hz, harmonic)
    #ft_filtered_pred = filter_fft(freqs_pred, ft_pred, band_size, Hz, harmonic)
    
    log_ft_filtered = np.log10(np.abs(ft_filtered))
    #log_ft_filtered_pred = np.log10(np.abs(ft_filtered_pred))
    
    ax1.plot(freqs, log_ft_filtered, color = "blue", label="real")
    #ax1.plot(freqs_pred, log_ft_filtered_pred, color = "red", label="pred")
    ax1.set_xlim(harmonic*int(Hz)-harmonic*int(Hz)*0.03, harmonic*int(Hz)+harmonic*int(Hz)*0.03)
    ax1.set_title(f"Harmonic {harmonic}")

    handles_1, labels_1 = ax1.get_legend_handles_labels()

    ax2 = axs2[harmonic - 1]

    inverseft=np.fft.ifft(ft_filtered)
    #inverseft_pred=np.fft.ifft(ft_filtered_pred)

    ax2.plot(time_data, inverseft, color = "blue", label="real")
    #ax2.plot(time_data, inverseft_pred, color = "red", label="pred")
    ax2.set_title(f"Harmonic {harmonic}")

    handles_2, labels_2 = ax2.get_legend_handles_labels()

    ax3 = axs3[harmonic - 1]
    ax3.plot(voltage, inverseft, color = "blue", label="real")
    #ax3.plot(voltage, inverseft_pred, color = "red", label="pred")
    ax3.set_title(f"Harmonic {harmonic}")

    handles_3, labels_3 = ax3.get_legend_handles_labels()


    log_ft_filtered[~np.isfinite(log_ft_filtered)] = 0
    #log_ft_filtered_pred[~np.isfinite(log_ft_filtered_pred)] = 0

fig1.legend(handles_1, labels_1, loc='lower right')
fig1.supxlabel("Frequency (Hz)")
fig1.supylabel("Log10 abs(ft)")
fig1.suptitle(f"File {Hz}Hz harmonics in freq domain")
fig1.tight_layout()
fig1.savefig(os.path.join(output_filepath, f"FTacV-harmonics-freq-3x3.png"))
plt.close(fig1.figure)

fig2.legend(handles_2, labels_2, loc='lower right')
fig2.supxlabel("Time")
fig2.supylabel("Current")
fig2.suptitle(f"File {Hz}Hz harmonics in time domain")
fig2.tight_layout()
fig2.savefig(os.path.join(output_filepath, f"FTacV-harmonics-time-3x3.png"))
plt.close(fig2.figure)

fig3.legend(handles_3, labels_3, loc='lower right')
fig3.supxlabel("Voltage")
fig3.supylabel("Current")
fig3.suptitle(f"File {Hz}Hz harmonics in voltage domain")
fig3.tight_layout()
fig3.savefig(os.path.join(output_filepath, f"FTacV-harmonics-voltage-3x3.png"))
plt.close(fig3.figure) """