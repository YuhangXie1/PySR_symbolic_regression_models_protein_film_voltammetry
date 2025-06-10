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
    """Returns the mean squared error between a y value (real) and a predicted y value (prediction)."""
    mse = sum((y_pred - y)**2)
    return mse


def load_data(Hz):
    """
    Returns numpy arrays of time_data, voltage, current and frequency for a set of data files
    """
    #loading data
    loc=r".\Data_for_eq_learning"
    #files=os.listdir(loc)

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


    #adding details to metadata
    Path(os.path.join(output_filepath)).mkdir(parents = True, exist_ok = True)
    with open(os.path.join(output_filepath, "metadata.txt"), "a") as metadata:
        metadata.write(f"time data: slice start: {slice_start}, slice end: {slice_end} \n")
    
    return time_data, voltage, current, freq


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

    return voltage_equation, [A, w, p, c]


def model(time_data, coeff_array, dv_dt, current, ID):
    """
    Runs a PySR model to fit ([voltage, dv_dt], current) and returns the best model.
    """

    #X = np.array([voltage, voltage**2, voltage**3, dv_dt, dv_dt**2, dv_dt**3]).T
    A = np.full((1,len(time_data)), coeff_array[0])[0]
    w = np.full((1,len(time_data)), coeff_array[1])[0]
    p = np.full((1,len(time_data)), coeff_array[2])[0]
    c = np.full((1,len(time_data)), coeff_array[3])[0]

    X = np.array([A, w, p, c,
                  np.sin(time_data * w), np.cos(time_data * w),
                    np.sin(time_data * w *2), np.cos(time_data * w * 2),
                    np.sin(time_data * w *3), np.cos(time_data * w * 3),
                    np.sin(time_data * w *4), np.cos(time_data * w * 4),
                    np.sin(time_data * w *5), np.cos(time_data * w * 5),
                  ]).T
    Y = np.array(current).reshape(-1,1)

    #pysr model definition
    model = PySRRegressor(
        maxsize=30,
        niterations=100,
        batching= True,
        binary_operators=["+","*"],
        elementwise_loss="loss(prediction, target) = (prediction - target)^2",
    )

    model.fit(X,Y)
    predicted_current = model.predict(X)

    #writing model parameters to metadata
    Path(os.path.join(output_filepath, str(ID))).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(output_filepath, str(ID), "metadata.txt"), "a") as metadata:
        metadata.write(f"model run ID: {model.run_id_}")
        metadata.write('''
    A = np.full((1,len(time_data), coeff_array[0]))
    w = np.full((1,len(time_data), coeff_array[1]))
    p = np.full((1,len(time_data), coeff_array[2]))
    c = np.full((1,len(time_data), coeff_array[3]))
    X = np.array([A, w, p, c,
                  np.sin(time_data * w), np.cos(time_data * w),
                    np.sin(time_data * w *2), np.cos(time_data * w * 2),
                    np.sin(time_data * w *3), np.cos(time_data * w * 3),
                    np.sin(time_data * w *4), np.cos(time_data * w * 4),
                    np.sin(time_data * w *5), np.cos(time_data * w * 5),
                  ]).T
    Y = np.array(current).reshape(-1,1)
                       
        model = PySRRegressor(
        maxsize=30,
        niterations=100,
        batching= True,
        binary_operators=["+","*"],
        elementwise_loss="loss(prediction, target) = (prediction - target)^2",
    )

                       \n''')
        
    #printing and writing to file
    model.equations_.to_csv(os.path.join(output_filepath, str(ID), "equations.csv"))
    

    with open(os.path.join(output_filepath, "summary_current_model.csv"), "a", newline='') as file:
        best = model.get_best()
        expanded_form = expand(sympify(str(model.sympy())))
        x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13 = symbols("x0 x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13")
        A,w,p,c,t = symbols("A,w,p,c,t")

        substituted_form = expand(expanded_form.subs(
                [(x0,A), (x1,w), (x2,p), (x3,c),
                  (x4,sp.sin(t * w)), (x5,sp.cos(t * w)),
                    (x6,sp.sin(t * w * 2)), (x7,sp.cos(t * w * 2)),
                    (x8,sp.sin(t * w * 3)), (x9,sp.cos(t * w * 3)),
                    (x10,sp.sin(t * w * 4)), (x11,sp.cos(t * w * 4)),
                    (x12,sp.sin(t * w * 5)), (x13,sp.cos(t * w * 5)),
                  ]))

        
        writer = csv.writer(file)
        writer.writerow([
                        ID,
                        best.loss,
                        best.complexity,
                        calculate_MSE(current, model.predict(X)),
                        expanded_form,
                        substituted_form,
                        ])
        
    #current time graph
    fig, axs = plt.subplots(3)
    axs[0].plot(time_data, current, label = "data", color = "orange")
    axs[0].plot(time_data, predicted_current, label = "prediction", color = "magenta")
    axs[0].set_xlabel("time")
    axs[0].set_ylabel("current")
    axs[0].set_title(f"Repeat {ID}. Current vs time. Whole. ")
    axs[0].legend()

    slice_start = 10550
    slice_end = 20000
    axs[1].plot(time_data[slice_start:slice_end], current[slice_start:slice_end], label = "data", color = "orange")
    axs[1].plot(time_data[slice_start:slice_end], predicted_current[slice_start:slice_end], label = "pred", color = "magenta", linestyle = "dotted")
    axs[1].set_xlabel("time")
    axs[1].set_ylabel("current")
    axs[1].set_title(f"Repeat {ID}. Current vs time. Slice {slice_start}-{slice_end}.")
    axs[1].legend()

    slice_start = 30550
    slice_end = 40000
    axs[2].plot(time_data[slice_start:slice_end], current[slice_start:slice_end], label = "data", color = "orange")
    axs[2].plot(time_data[slice_start:slice_end], predicted_current[slice_start:slice_end], label = "pred", color = "magenta", linestyle = "dotted")
    axs[2].set_xlabel("time")
    axs[2].set_ylabel("current")
    axs[2].set_title(f"Repeat {ID}. Current vs time. Slice {slice_start}-{slice_end}.")
    axs[2].legend()

    fig.tight_layout()
    plt.savefig(os.path.join(output_filepath, str(ID), f"fig{ID}-current-time.png"))

    #current dV/dt graph
    fig, axs = plt.subplots()
    axs.plot(dv_dt, current, label = "data", color = "orange")
    axs.plot(dv_dt, predicted_current, label = "prediction", color = "magenta")
    axs.set_xlabel("dV/dt")
    axs.set_ylabel("current")
    axs.set_title(f"Repeat {ID}. Current vs dV/dt.")
    axs.legend()

    fig.tight_layout()
    plt.savefig(os.path.join(output_filepath, str(ID), f"fig{ID}-current-dv_dt.png"))

    return model



### main ###

#input variables
number_of_repeats = 10
Hz = 36
output_filepath = rf"results/20250610-fit-workflow-fourier-basis/{Hz}"

Path(output_filepath).mkdir(parents=True, exist_ok=True)
with open(os.path.join(output_filepath, "summary_current_model.csv"), "a", newline='') as file:
    writer = csv.writer(file)
    writer.writerow([
                "ID",
                "loss",
                "complexity",
                "MSE",
                "picked_equation",
                "substituted_form",
                ])
    
with open(os.path.join(output_filepath, "summary_voltage_model.csv"), "a", newline='') as file:
    writer = csv.writer(file)
    writer.writerow([
                "voltage_equation",
                "dv_dt_equation",
                "coeff A",
                "coeff w",
                "coeff p",
                "coeff c",
                ])

time_data, voltage, current, freq = load_data(Hz)

x0 = Symbol("x0")
voltage_eqn, coeff_array = fit_sin(time_data, voltage)
dv_dt_func = diff(voltage_eqn, x0)
dv_dt_lambda = lambdify(x0,dv_dt_func)
dv_dt = dv_dt_lambda(time_data)

with open(os.path.join(output_filepath, "summary_voltage_model.csv"), "a", newline='') as file:
    writer = csv.writer(file)
    writer.writerow([
                voltage_eqn,
                dv_dt_func,
                coeff_array[0],
                coeff_array[1],
                coeff_array[2],
                coeff_array[3],
                ])
    
for i in range(0,number_of_repeats):

    start_time = time.time()
    select_model = model(time_data, coeff_array, dv_dt, current, i)
    print (expand(select_model.sympy()))
    end_time = time.time()

    time_elapsed = str(datetime.timedelta(seconds = end_time - start_time))
    with open(os.path.join(output_filepath, "metadata.txt"), "a") as metadata:
        metadata.write(f"model {i} time taken = {time_elapsed} hr:min:sec \n")