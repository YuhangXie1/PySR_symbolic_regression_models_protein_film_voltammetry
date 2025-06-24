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
    data_volt = pd.read_csv(f"Data_for_eq_learning/{Hz}_Hz_2_cv_voltage", sep="\t", names = ["time","voltage"])
    data_amp = pd.read_csv(f"Data_for_eq_learning/{Hz}_Hz_2_cv_current", sep="\t", names = ["time","current"])
    
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
files_freq = [9, 36, 45, 54, 63, 72, 81, 90, 99]

load_filepath = rf"results\20250620-multi-fit-workflow-9\time-shift-normalised_dv_dt_aval"
output_filepath = rf"results/20250624-normalised-no-div/fft"
Path(output_filepath).mkdir(parents=True, exist_ok=True)

combined_data_array = combine_data(files_freq)
time_data = combined_data_array[:,0]
x_dat = combined_data_array[:,1]
dx_dat = combined_data_array[:,2]
current = combined_data_array[:,3]
A_dat = combined_data_array[:,4]
w_dat = combined_data_array[:,5]
p_dat = combined_data_array[:,6]
c_dat = combined_data_array[:,7]
A_val_dat = combined_data_array[:,8]
p_val_dat = combined_data_array[:,9]

current_summary_file = pd.read_csv(os.path.join(load_filepath, "summary_current_model.csv"), header=0)
current_equations = current_summary_file["substituted_form"]
x, dx, A, w, p, c, A_val, p_val = symbols("x dx A w p c A_val p_val")


for num, eqn in enumerate(current_equations):
    current_eqn_lambda = lambdify([x, dx, A, w, p, c, A_val, p_val], eqn)
    current_pred = current_eqn_lambda(x_dat, dx_dat, A_dat, w_dat, p_dat, c_dat, A_val_dat, p_val_dat)





