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
data = pd.read_csv(f"Data_for_eq_learning/20250609/20250609-blank-PSV_36Hz.csv", header = 0)

#slicing data to remove start and end noise. Putting data into tuples
slice_start = 0
slice_end = -1
time_data_new = np.array(data["x"].iloc[slice_start:slice_end])
voltage_new = np.array(data["z"].iloc[slice_start:slice_end])
current_new = np.array(data["y"].iloc[slice_start:slice_end])

time_data_old, voltage_old, current_old, freq, slice = load_data(36)


""" fig, axs = plt.subplots()
axs.plot(voltage_old, current_old, label = "Old", color = "blue")
axs.plot(voltage_new, current_new, label = "Today", color = "red")
axs.set_xlabel("Voltage")
axs.set_ylabel("Current")
axs.legend()
fig.tight_layout() """

sinusoid_new = 0.3*np.sin(2*np.pi*36*time_data_new - np.pi/2) - 0.05
sinusoid_old = 0.3*np.sin(2*np.pi*36*time_data_old - np.pi/2) - 0.05

voltage_pred_old = 0.299894563526984*np.sin(228.449287868794*time_data_old - 1.55129181346913) - 0.0496859862566539
sinusoid_old_2 = 0.3*np.sin(228.449287868794*time_data_old - np.pi/2) - 0.05
sinusoid_old_3 = 0.3*np.sin(228.449287868794*time_data_old - 1.55129181346913) - 0.05
sinusoid_old_4 = 0.3*np.sin((np.pi*2*36+2.2546168103288835)*time_data_old - (np.pi/2 -0.019504513325766526)) - 0.05


print(228.449287868794 - np.pi*2*36)
print(1.55129181346913 - np.pi/2)

fig2, axs2 = plt.subplots()
#axs2.plot(time_data_old, voltage_old, label = "Old", color = "blue")
#axs2.plot(time_data_new, voltage_new, label = "Today", color = "red")
#axs2.plot(time_data_old, sinusoid_old, label = "Artificial", color = "green", linestyle = "dashed")
#axs2.plot(time_data_new, sinusoid_new, label = "Artificial", color = "green", linestyle = "dashed")
axs2.plot(time_data_old, voltage_old, label = "real", color = "blue")
axs2.plot(time_data_old, voltage_pred_old, label = "pred", color = "red", linestyle = "dotted")
axs2.set_xlabel("Time")
axs2.set_ylabel("Voltage")
axs2.legend()
fig2.tight_layout()

""" fig3, axs3 = plt.subplots()
axs3.plot(time_data_old, current_old, label = "Old", color = "blue")
axs3.plot(time_data_new, current_new, label = "Today", color = "red")
axs3.set_xlabel("Time")
axs3.set_ylabel("Current")
axs3.legend()
fig3.tight_layout() """



fig4, axs4 = plt.subplots()
axs4.plot(voltage_old, voltage_pred_old, label = "old", color = "blue")
axs4.plot(voltage_old, sinusoid_old_4, label = "pred", color = "red", linestyle = "dotted")
axs4.set_xlabel("Voltage real")
axs4.set_ylabel("Voltage pred")
axs4.legend()
fig4.tight_layout()

plt.show()