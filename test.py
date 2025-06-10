import numpy as np
from sympy import Symbol, sympify, symbols, diff, Wild
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
    loc=r".\Data_for_eq_learning"

    #loading data
    data_volt = pd.read_csv(f"Data_for_eq_learning/{Hz}_Hz_2_cv_voltage", sep="\t", names = ["time","voltage"])
    data_amp = pd.read_csv(f"Data_for_eq_learning/{Hz}_Hz_2_cv_current", sep="\t", names = ["time","current"])
    data_combined = data_volt.join(data_amp["current"])
    data_combined.insert(0, "Freq", Hz)

    #slicing data to remove start and end noise. Putting data into tuples
    slice_start = 550
    slice_end = -100
    time_data = np.array(data_combined["time"].iloc[slice_start:slice_end])
    voltage = np.array(data_combined["voltage"].iloc[slice_start:slice_end])
    current = np.array(data_combined["current"].iloc[slice_start:slice_end])
    freq = np.array(data_combined["Freq"].iloc[slice_start:slice_end])

    return time_data, voltage, current, freq, [slice_start, slice_end]



#####
files_freq = [36, 45]

combined_data_array = np.empty((1,4))

for Hz in files_freq:
    time_data, voltage, current, freq, slice_array = load_data(Hz)
    combined_data_array = np.concatenate([combined_data_array, np.array([time_data, voltage, current, freq]).T],0)

combined_data_array = np.delete(combined_data_array, (0), axis=0)
#print(combined_data_array)





#loading data
data = pd.read_csv("Data_for_eq_learning/20250610/20250610-blank-PSV-36Hz.csv", header = 0)

#slicing data to remove start and end noise. Putting data into tuples
slice_start = 0
slice_end = -1
time_data_new = np.array(data["x"].iloc[slice_start:slice_end])
voltage_new = np.array(data["z"].iloc[slice_start:slice_end])
current_new = np.array(data["y"].iloc[slice_start:slice_end])

data = pd.read_csv("Data_for_eq_learning/20250610/20250610-NimA-PSV-36Hz.csv", header = 0)

#slicing data to remove start and end noise. Putting data into tuples
slice_start = 0
slice_end = -1
time_data_new_nima = np.array(data["x"].iloc[slice_start:slice_end])
voltage_new_nima = np.array(data["z"].iloc[slice_start:slice_end])
current_new_nima = np.array(data["y"].iloc[slice_start:slice_end])

time_data_old, voltage_old, current_old, freq, slice = load_data(36)


sinusoid_new = 0.3*np.sin(2*np.pi*36*time_data_new - np.pi/2) - 0.05
sinusoid_old = 0.3*np.sin(2*np.pi*36*time_data_old - np.pi/2) - 0.05

voltage_pred_old = 0.299894563526984*np.sin(228.449287868794*time_data_old - 1.55129181346913) - 0.0496859862566539
sinusoid_old_2 = 0.3*np.sin(228.449287868794*time_data_old - np.pi/2) - 0.05
sinusoid_old_3 = 0.3*np.sin(228.449287868794*time_data_old - 1.55129181346913) - 0.05
sinusoid_old_4 = 0.3*np.sin((np.pi*2*36+2.2546168103288835)*time_data_old - (np.pi/2 -0.019504513325766526)) - 0.05
sinusoid_old_5 = 0.3*np.sin((np.pi*2*36)*time_data_old - (np.pi/2)) - 0.05

voltage_pred_new = 0.299791538901415*np.sin(226.203834030565*time_data_new - 1.58242902181405) - 0.050229948540937
sinusoid_new_2 = 0.3*np.sin(226.203834030565*time_data_new - 1.58242902181405) - 0.05
sinusoid_new_3 = 0.3*np.sin(226.203834030565*time_data_new - np.pi/2) - 0.05
sinusoid_new_4 = 0.3*np.sin((np.pi*2*36)*time_data_new - (np.pi/2)) - 0.05
sinusoid_new_5 = 0.3*np.sin((np.pi*2*36+2.2546168103288835)*time_data_new - (np.pi/2 -0.019504513325766526)) - 0.05

#print(np.pi*2*36 - 226.203834030565)
#print(np.pi/2 - 1.58242902181405)

#-0.009162972099886701
#-0.011632695019153427
#36.358833410141926

print(228.449287868794 / (np.pi * 2))
print(1.55129181346913 / (np.pi))

""" fig, axs = plt.subplots(2,2)

axs[0,0].plot(voltage_old, voltage_old, label = "real", color = "blue")
axs[0,0].plot(voltage_old, voltage_pred_old, label = "pred", color = "red", linestyle = "dotted")
axs[0,0].set_title("Learned eqn: \n 0.299894563526984*np.sin(228.449287868794*time_data_old - 1.55129181346913) - 0.0496859862566539")

axs[0,1].plot(voltage_old, voltage_old, label = "real", color = "blue")
axs[0,1].plot(voltage_old, sinusoid_old_3, label = "pred", color = "red", linestyle = "dotted")
axs[0,1].set_title("Eqn: \n 0.3*np.sin(228.449287868794*time_data_old - 1.55129181346913) - 0.05")

axs[1,0].plot(voltage_old, voltage_old, label = "real", color = "blue")
axs[1,0].plot(voltage_old, sinusoid_old_2, label = "pred", color = "red", linestyle = "dotted")
axs[1,0].set_title("Eqn: \n 0.3*np.sin(228.449287868794*time_data_old - np.pi/2) - 0.05")

axs[1,1].plot(voltage_old, voltage_old, label = "real", color = "blue")
axs[1,1].plot(voltage_old, sinusoid_old_5, label = "pred", color = "red", linestyle = "dotted")
axs[1,1].set_title("Eqn: \n 0.3*np.sin((np.pi*2*36)*time_data_old - (np.pi/2)) - 0.05")


fig.supxlabel("Voltage pred")
fig.supylabel("Voltage real")
fig.tight_layout()
plt.show() """

""" fig, axs = plt.subplots(2,2)

axs[0,0].plot(voltage_new, voltage_new, label = "real", color = "blue")
axs[0,0].plot(voltage_new, voltage_pred_new, label = "pred", color = "red", linestyle = "dotted")
axs[0,0].set_title("Learned eqn: \n 0.299791538901415*np.sin(226.203834030565*time_data_new - 1.58242902181405) - 0.050229948540937")

axs[0,1].plot(voltage_new, voltage_new, label = "real", color = "blue")
axs[0,1].plot(voltage_new, sinusoid_new_2, label = "pred", color = "red", linestyle = "dotted")
axs[0,1].set_title("Eqn: \n 0.3*np.sin(226.203834030565*time_data_new - 1.58242902181405) - 0.05")

axs[1,0].plot(voltage_new, voltage_new, label = "real", color = "blue")
axs[1,0].plot(voltage_new, sinusoid_new_3, label = "pred", color = "red", linestyle = "dotted")
axs[1,0].set_title("Eqn: \n 0.3*np.sin(226.203834030565*time_data_new - np.pi/2) - 0.05")

axs[1,1].plot(voltage_new, voltage_new, label = "real", color = "blue")
axs[1,1].plot(voltage_new, sinusoid_new_4, label = "pred", color = "red", linestyle = "dotted")
axs[1,1].set_title("Eqn: \n 0.3*np.sin((np.pi*2*36)*time_data_old - (np.pi/2)) - 0.05")


fig.supxlabel("Voltage pred")
fig.supylabel("Voltage real")
fig.tight_layout()
plt.show() """


""" fig, axs = plt.subplots()
axs.plot(time_data_old, voltage_old, label = "real", color = "blue")
axs.plot(time_data_old, sinusoid_old, label = "sinusoid", color = "red", linestyle = "dotted")
axs.set_xlabel("time")
axs.set_ylabel("Voltage")
axs.legend()
fig.tight_layout()
plt.show() """
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
freqs, ft = fourier_transform(current_new, time_data_new)
freqs_pred, ft_pred = fourier_transform(current_new_nima, time_data_new)

""" ax1.plot(freqs, log_ft_filtered, color = "blue", label="real")
ax1.plot(freqs_pred, log_ft_filtered_pred, color = "red", label="pred")
ax1.set_xlim(harmonic*int(Hz)-harmonic*int(Hz)*0.03, harmonic*int(Hz)+harmonic*int(Hz)*0.03)
ax1.set_title(f"Harmonic {harmonic}") """


fig, axs = plt.subplots()

axs.plot(voltage_new, current_new, label = "blank", color = "blue")
axs.plot(voltage_new_nima, current_new_nima, label = "NimA", color = "red")
axs.set_xlabel("Voltage")
axs.set_ylabel("Current")
axs.legend()
fig.tight_layout()


fig2, axs2 = plt.subplots()

axs2.plot(freqs, np.log10(ft**2), color = "blue", label="blank")
axs2.plot(freqs_pred, np.log10(ft_pred**2), color = "red", label="NimA")
axs2.set_xlim(0,400)
axs2.set_ylim(-8, 2)
axs2.set_xlabel("Frequency")
axs2.set_ylabel("Log10 ft^2")
axs2.legend()
fig2.tight_layout()


harmonic = 1
ft_filtered = filter_fft(freqs, ft, band_size, Hz, harmonic)
ft_filtered_pred = filter_fft(freqs_pred, ft_pred, band_size, Hz, harmonic)

#log_ft_filtered = np.log10(np.abs(ft_filtered))
#log_ft_filtered_pred = np.log10(np.abs(ft_filtered_pred))

inverseft=np.fft.ifft(ft_filtered)
inverseft_pred=np.fft.ifft(ft_filtered_pred)

fig3, axs3 = plt.subplots()
axs3.plot(time_data_new, inverseft, color = "blue", label="blank")
axs3.plot(time_data_new, inverseft_pred, color = "red", label="NimA")
axs3.legend()
axs3.set_xlabel("Time")
axs3.set_ylabel("Current")
axs3.set_title(f"File {Hz}Hz harmonic {harmonic} in time domain")
fig3.tight_layout()

harmonic = 7
ft_filtered = filter_fft(freqs, ft, band_size, Hz, harmonic)
ft_filtered_pred = filter_fft(freqs_pred, ft_pred, band_size, Hz, harmonic)

#log_ft_filtered = np.log10(np.abs(ft_filtered))
#log_ft_filtered_pred = np.log10(np.abs(ft_filtered_pred))

inverseft=np.fft.ifft(ft_filtered)
inverseft_pred=np.fft.ifft(ft_filtered_pred)

fig4, axs4 = plt.subplots()
axs4.plot(time_data_new, inverseft, color = "blue", label="blank")
axs4.plot(time_data_new, inverseft_pred, color = "red", label="NimA")
axs4.legend()
axs4.set_xlabel("Time")
axs4.set_ylabel("Current")
axs4.set_title(f"File {Hz}Hz harmonic {harmonic} in time domain")
fig4.tight_layout()

plt.show()