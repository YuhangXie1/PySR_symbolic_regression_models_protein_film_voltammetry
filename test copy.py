import numpy as np
from sympy import Symbol, sympify, symbols, diff
import sympy as sp
from sympy.utilities.lambdify import lambdify
from scipy.optimize import curve_fit
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import csv
import os

def calculate_MSE(y, y_pred):
    """Returns the mean squared error between a y value (real) and a predicted y value (prediction)."""
    mse = sum((y_pred - y)**2)
    return mse

def calculate_MSE_complex(y, y_pred):
    """Returns the mean squared error between a complex y value and a predicted complex y value."""
    mse = sum(np.abs(y_pred - y)**2)
    return mse

Hz = 36
data_volt = pd.read_csv(f"Data_for_eq_learning/{Hz}_Hz_2_cv_voltage", sep="\t", names = ["time","voltage"])
data_amp = pd.read_csv(f"Data_for_eq_learning/{Hz}_Hz_2_cv_current", sep="\t", names = ["time","current"])
data_combined = data_volt.join(data_amp["current"])

#slicing data to remove start and end noise. Putting data into tuples
slice_start = 550
slice_end = -100
time_data = np.array(data_combined["time"].iloc[slice_start:slice_end])
voltage = np.array(data_combined["voltage"].iloc[slice_start:slice_end])
current = np.array(data_combined["current"].iloc[slice_start:slice_end])

x, dx, x0 = symbols("x dx x0")
voltage_eqn = sympify("0.299894563526963*sin(228.4492878688*x0 - 1.55129181348585) - 0.0496859862564122")
dv_dt_func = diff(voltage_eqn, x0)
dv_dt_lambda = lambdify(x0,dv_dt_func)
dv_dt = dv_dt_lambda(time_data)

current_pred_eqn = 1.83479321254227e-6*dx*x**3 - 7.77746142237813e-7*dx*x**2 - 7.77746142237813e-7*dx*x + 7.5642693e-6*dx + 0.00060018763*x**2 + 0.0010604324*x + 2.4237355e-5
current_pred_lambda = lambdify([x, dx], current_pred_eqn)
current_pred = current_pred_lambda(voltage, dv_dt)






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

band_size=0.1
desired_harmonic=1
freqs, ft = fourier_transform(current, time_data)
freqs_pred, ft_pred = fourier_transform(current_pred, time_data)

ft_filtered = filter_fft(freqs, ft, band_size, Hz, desired_harmonic)
ft_filtered_pred = filter_fft(freqs_pred, ft_pred, band_size, Hz, desired_harmonic)

inverseft=np.fft.ifft(ft_filtered)
inverseft_pred=np.fft.ifft(ft_filtered_pred)

log_ft_filtered = np.log10(np.abs(ft_filtered))
log_ft_filtered_pred = np.log10(np.abs(ft_filtered_pred))

data = np.array([freqs,log_ft_filtered,log_ft_filtered_pred]).T
data_new = np.array([row for row in data if np.isfinite(row[1]) and np.isfinite(row[2])])

fig, ax1 = plt.subplots()
ax1.scatter(data_new[:,0], data_new[:,1], color = "blue", label="real")
ax1.scatter(data_new[:,0], data_new[:,2], color = "red", label="pred")
#ax1.set_xlim(harmonic*int(Hz)-harmonic*int(Hz)*0.03, harmonic*int(Hz)+harmonic*int(Hz)*0.03)
ax1.set_title(f"Harmonic {desired_harmonic}")
