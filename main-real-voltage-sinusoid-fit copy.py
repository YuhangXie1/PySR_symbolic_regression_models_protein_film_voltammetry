from pysr import PySRRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import datetime
from pathlib import Path
import csv
from sympy import Symbol, sympify, expand, symbols, diff
from sympy.utilities.lambdify import lambdify
from scipy.optimize import curve_fit
import sympy as sp

def calculate_MSE(y, y_pred):

    mse = sum((y_pred - y)**2)
    print(f"MSE:{mse}")

    return mse

def loading_data():
    #loading data
    loc=r".\Data_for_eq_learning"
    files=os.listdir(loc)

    data_volt = pd.read_csv("Data_for_eq_learning/36_Hz_2_cv_voltage", sep="\t", names = ["time","voltage"])
    data_amp = pd.read_csv("Data_for_eq_learning/36_Hz_2_cv_current", sep="\t", names = ["time","current"])
    data_combined_9 = data_volt.join(data_amp["current"])
    data_combined_9.insert(0, "Freq", 9)

    #slicing data to remove start and end noise. Putting data into tuples
    slice_start = 550
    slice_end = -100
    time_data = np.array(data_combined_9["time"].iloc[slice_start:slice_end])
    voltage = np.array(data_combined_9["voltage"].iloc[slice_start:slice_end])
    current = np.array(data_combined_9["current"].iloc[slice_start:slice_end])
    freq = np.array(data_combined_9["Freq"].iloc[slice_start:slice_end])

    return time_data, voltage

def fit_sin(tt, yy, ID):
    '''
    Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"
    Obtained from https://stackoverflow.com/questions/16716302/how-do-i-fit-a-sine-curve-to-my-data-with-pylab-and-numpy, user unsym, Feb 19, 2017

    '''
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


    x0 = Symbol("x0")
    equation = A*sp.sin(w*x0 + p) + c
    equation_func = lambdify(x0, equation)
    y_pred = equation_func(tt)

    Path(os.path.join(output_filepath, str(ID))).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(output_filepath, "summary.csv"), "a", newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
                        ID,
                        calculate_MSE(yy, y_pred),
                        equation,
                        A,
                        w,
                        p,
                        c,
                        ])
        
    #writing model parameters to metadata
    with open(os.path.join(output_filepath, str(ID), "metadata.txt"), "a") as metadata:
        metadata.write("params = curve_fit(function, x, y)\n")
        metadata.write("function = A * np.sin(B * x + C) + D\n")

    generate_plots(tt, yy, y_pred, ID)

    return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1./f, "fitfunc": fitfunc, "maxcov": np.max(pcov), "rawres": (guess,popt,pcov)}

def generate_plots(time_data, voltage, predicted_voltage, ID):

    fig, axs = plt.subplots()
    slice_start = 0
    slice_end = -1
    axs.plot(time_data[slice_start:slice_end], voltage[slice_start:slice_end], label = "data", color = "cyan")
    axs.plot(time_data[slice_start:slice_end], predicted_voltage[slice_start:slice_end], label = "pred", color = "red", linestyle = "dotted")
    axs.set_xlabel("time")
    axs.set_ylabel("voltage")
    axs.set_title(f"Repeat {ID} voltage vs time")
    axs.legend()
    fig.tight_layout()
    plt.savefig(os.path.join(output_filepath, str(ID), f"fig{ID}-current-time-whole.png"))

    fig, axs = plt.subplots()
    slice_start = 10550
    slice_end = 20000
    axs.plot(time_data[slice_start:slice_end], voltage[slice_start:slice_end], label = "data", color = "cyan")
    axs.plot(time_data[slice_start:slice_end], predicted_voltage[slice_start:slice_end], label = "pred", color = "red", linestyle = "dotted")
    axs.set_xlabel("time")
    axs.set_ylabel("voltage")
    axs.set_title(f"Repeat {ID} voltage vs time")
    axs.legend()
    fig.tight_layout()
    plt.savefig(os.path.join(output_filepath, str(ID), f"fig{ID}-voltage-time-slice-1.png"))

    fig, axs = plt.subplots()
    slice_start = 30550
    slice_end = 40000
    axs.plot(time_data[slice_start:slice_end], voltage[slice_start:slice_end], label = "data", color = "cyan")
    axs.plot(time_data[slice_start:slice_end], predicted_voltage[slice_start:slice_end], label = "pred", color = "red", linestyle = "dotted")
    axs.set_xlabel("time")
    axs.set_ylabel("voltage")
    axs.set_title(f"Repeat {ID} voltage vs time")
    axs.legend()
    fig.tight_layout()
    plt.savefig(os.path.join(output_filepath, str(ID), f"fig{ID}-voltage-time-slice-2.png"))

### main ###

#input variables
number_of_repeats = 10

output_filepath = rf"results/20250512-real-curve-fit-voltage-time-36/test-1"
Path(output_filepath).mkdir(parents=True, exist_ok=True)
with open(os.path.join(output_filepath, "summary.csv"), "a", newline='') as file:
    writer = csv.writer(file)
    writer.writerow([
                "ID",
                "MSE",
                "picked_equation",
                "Coeff A",
                "Coeff B",
                "Coeff C",
                "Coeff D",
                ])

for i in range(0,number_of_repeats):
    time_data, voltage = loading_data()
    fit_sin(time_data, voltage, i)

