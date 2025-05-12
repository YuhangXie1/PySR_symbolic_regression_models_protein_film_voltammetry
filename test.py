import numpy as np
from sympy import Symbol, sympify
import sympy as sp
from sympy.utilities.lambdify import lambdify
from scipy.optimize import curve_fit
import pandas as pd

x = np.array(range(0,100))
y = 3 * np.sin(2*x + 4) - 2

def sinusoidal_fit(x, A, B, C, D):
    return A * np.sin(B * x + C) + D

data_volt = pd.read_csv("Data_for_eq_learning/9_Hz_2_cv_voltage", sep="\t", names = ["time","voltage"])
data_amp = pd.read_csv("Data_for_eq_learning/9_Hz_2_cv_current", sep="\t", names = ["time","current"])
data_combined_9 = data_volt.join(data_amp["current"])
data_combined_9.insert(0, "Freq", 9)

slice_start = 550
slice_end = -100
time_data = np.array(data_combined_9["time"].iloc[slice_start:slice_end])
voltage = np.array(data_combined_9["voltage"].iloc[slice_start:slice_end])
current = np.array(data_combined_9["current"].iloc[slice_start:slice_end])
freq = np.array(data_combined_9["Freq"].iloc[slice_start:slice_end])

""" x0 = Symbol("x0")
equation = sympify(params[0]*sp.sin(params[1]*x0 + params[2]) + params[3])
func = lambdify(x0, equation)
print(equation)

x = np.array([0,1,2,3,4])

y_pred = func(x) """




def fit_sin(tt, yy):
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
    return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1./f, "fitfunc": fitfunc, "maxcov": np.max(pcov), "rawres": (guess,popt,pcov)}


print(fit_sin(time_data,voltage))