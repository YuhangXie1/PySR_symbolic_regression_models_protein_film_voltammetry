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
    mse = sum((y_pred - y)**2)/len(y)
    return mse

def calculate_MSE_complex(y, y_pred):
    """Returns the mean squared error between a complex y value and a predicted complex y value."""
    mse = sum(np.abs(y_pred - y)**2)/len(y)
    return mse

def load_data(Hz):
    """
    Returns numpy arrays of time_data, voltage, current and frequency for a set of data files
    """
    #loading data
    data_volt = pd.read_csv(f"Data_for_eq_learning/{Hz}_Hz_1_cv_voltage", sep="\t", names = ["time","voltage"])
    data_amp = pd.read_csv(f"Data_for_eq_learning/{Hz}_Hz_1_cv_current", sep="\t", names = ["time","current"])
    
    #data_amp = pd.read_csv(r"Data_for_eq_learning\FTacV_before_PSV_cv_current", sep="\t", names = ["time","current"])
    #data_volt = pd.read_csv(r"Data_for_eq_learning\FTacV_before_PSV_cv_voltage", sep="\t", names = ["time","voltage"])

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

def combine_data(files_freq):
    #find voltage eqn and stitch data together
    combined_data_array = np.empty((1,10))
    for Hz in files_freq:
        
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

        A = np.full((1,len(time_data)), coeff_array[0])[0]
        w = np.full((1,len(time_data)), coeff_array[1])[0]
        p = np.full((1,len(time_data)), coeff_array[2])[0]
        c = np.full((1,len(time_data)), coeff_array[3])[0]
        p_val = np.full((1,len(time_data)), p_val)[0]
        A_val = np.full((1,len(time_data)), A_val)[0]

        combined_data_array = np.concatenate([combined_data_array, np.array([time_data, voltage, dv_dt, current, A, w, p, c, A_val, p_val]).T],0)
    combined_data_array = np.delete(combined_data_array, (0), axis=0)
    return combined_data_array

####
#files_order = [9, 36, 45, 54, 63, 72, 81, 90, 99]
files_order = [9, 45, 54, 63, 72, 81, 90, 99]
load_filepath = r"results\20250620-multi-fit-workflow-9\time-shift-normalised_dv_dt_aval"

output_filepath = rf"results\20250620-multi-fit-workflow-9/file-predict/normalised-no-div-unknown-files/"
Path(output_filepath).mkdir(parents=True, exist_ok=True)

current_summary_file = pd.read_csv(os.path.join(load_filepath, "summary_current_model.csv"), header=0)
current_eqns = current_summary_file["substituted_form"]


with open(os.path.join(output_filepath, "summary.csv"), "a", newline='') as file:
    writer = csv.writer(file)
    writer.writerow([
                    "Hz_file",
                    "Eqn_num",
                    "MSE",
                    "predicting_eqn",
                    ])

for num, eqn in enumerate(current_eqns):

    for Hz in files_order:

        time_data, voltage, current, slice = load_data(Hz)
        voltage_eqn, dv_dt_eqn, coeff_array = fit_voltage_eqn(time_data, voltage)

        #current phase and amplitude data
        phase_data = pd.read_csv(rf"results\20250617-current-phase-original-set\summary.csv")
        p_val = np.array(phase_data.loc[phase_data["Hz"] == Hz]["phase"])[0]
        A_val = np.array(phase_data.loc[phase_data["Hz"] == Hz]["amplitude"])[0]
        w = coeff_array[1]

        #time shift t
        time_data = time_data + (p_val/w)

        #new time shifted voltage and dv_dt
        x0 = Symbol("x0")
        voltage_lambda = lambdify(x0,voltage_eqn)
        voltage = voltage_lambda(time_data)
        dv_dt_lambda = lambdify(x0,dv_dt_eqn)
        dv_dt = dv_dt_lambda(time_data)/w #normalise dV/dt

        #normalise current
        current = current/A_val

        #prepare variables for prediction of current
        A_dat = np.full((1,len(time_data)), coeff_array[0])[0]
        w_dat = np.full((1,len(time_data)), coeff_array[1])[0]
        p_dat = np.full((1,len(time_data)), coeff_array[2])[0]
        c_dat = np.full((1,len(time_data)), coeff_array[3])[0]
        p_val_dat = np.full((1,len(time_data)), p_val)[0]
        A_val_dat = np.full((1,len(time_data)), A_val)[0]
        x_dat = voltage
        dx_dat = dv_dt

        x, dx, A, w, p, c, A_val, p_val = symbols("x dx A w p c A_val p_val")
        current_eqn_lambda = lambdify([x, dx, A, w, p, c, A_val, p_val], eqn)
        current_pred = current_eqn_lambda(x_dat, dx_dat, A_dat, w_dat, p_dat, c_dat, A_val_dat, p_val_dat)

        MSE = calculate_MSE(current, current_pred)


        #current time graph
        fig, axs = plt.subplots(3)
        axs[0].plot(time_data, current, label = "data", color = "orange")
        axs[0].plot(time_data, current_pred, label = "pred", color = "magenta")
        axs[0].set_xlabel("time")
        axs[0].set_ylabel("current")
        axs[0].set_title(f"{Hz}Hz. Current vs time. Whole. ")
        axs[0].legend()

        slice_start = 10550
        slice_end = 20000
        axs[1].plot(time_data[slice_start:slice_end], current[slice_start:slice_end], label = "data", color = "orange")
        axs[1].plot(time_data[slice_start:slice_end], current_pred[slice_start:slice_end], label = "pred", color = "magenta", linestyle = "dotted")
        axs[1].set_xlabel("time")
        axs[1].set_ylabel("current")
        axs[1].set_title(f"{Hz}Hz. Current vs time. Slice {slice_start}-{slice_end}.")
        axs[1].legend()

        slice_start = 110550
        slice_end = 120000
        axs[2].plot(time_data[slice_start:slice_end], current[slice_start:slice_end], label = "data", color = "orange")
        axs[2].plot(time_data[slice_start:slice_end], current_pred[slice_start:slice_end], label = "pred", color = "magenta", linestyle = "dotted")
        axs[2].set_xlabel("time")
        axs[2].set_ylabel("current")
        axs[2].set_title(f"{Hz}Hz. Current vs time. Slice {slice_start}-{slice_end}.")
        axs[2].legend()
        fig.tight_layout()
        Path(os.path.join(output_filepath,f"Eqn-{num}")).mkdir(parents=True, exist_ok=True)
        plt.savefig(os.path.join(output_filepath, f"Eqn-{num}", f"{Hz}Hz-current-time.png"))
        plt.close(fig)

        #current by dv/dt graph
        fig, axs = plt.subplots()
        axs.plot(dv_dt, current, label = "real", color = "blue")
        axs.plot(dv_dt, current_pred, label = "pred", color = "red")
        axs.set_xlabel("dV/dt")
        axs.set_ylabel("current")
        axs.set_title(f"{Hz}Hz - dv_dt-current - MSE: {MSE}")
        axs.legend()
        fig.tight_layout()
        Path(os.path.join(output_filepath,f"Eqn-{num}")).mkdir(parents=True, exist_ok=True)
        plt.savefig(os.path.join(output_filepath, f"Eqn-{num}", f"{Hz}Hz-current-dv-dt.png"))
        plt.close(fig)

        with open(os.path.join(output_filepath, "summary.csv"), "a", newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                            Hz,
                            num,
                            MSE,
                            eqn,
                            ])
        
        #plotting fourier series
        band_size=0.1
        freqs, ft = fourier_transform(current, time_data)
        freqs_pred, ft_pred = fourier_transform(current_pred, time_data)

        #ploting all harmonics
        fig, axs = plt.subplots()
        axs.plot(freqs, np.log10(ft**2), color = "blue", label="real")
        axs.plot(freqs_pred, np.log10(ft_pred**2), color = "red", label="pred")
        axs.set_xlim(0,20*Hz)
        axs.set_ylim(0, 12)
        axs.set_xlabel("Frequency")
        axs.set_ylabel("Log10 ft^2")
        axs.set_title(f"File {Hz}Hz, Eqn {num} all harmonics")
        axs.legend()
        fig.tight_layout()
        plt.savefig(os.path.join(output_filepath, f"Eqn-{num}", f"{Hz}Hz-all-harmonic-freq.png"))
        plt.close(fig)