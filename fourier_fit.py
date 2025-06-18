from symfit import parameters, variables, sin, cos, Fit
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def load_data(Hz):
    """
    Returns numpy arrays of time_data, voltage, current and frequency for a set of data files
    """
    loc=r".\Data_for_eq_learning"

    #loading data
    data_volt = pd.read_csv(f"Data_for_eq_learning/{Hz}_Hz_2_cv_voltage", sep="\t", names = ["time","voltage"])
    data_amp = pd.read_csv(f"Data_for_eq_learning/{Hz}_Hz_2_cv_current", sep="\t", names = ["time","current"])

    #slicing data to remove start and end noise. Putting data into tuples
    slice_start = 550
    slice_end = -100
    time_data = np.array(data_volt["time"].iloc[slice_start:slice_end])
    voltage = np.array(data_volt["voltage"].iloc[slice_start:slice_end])
    current = np.array(data_amp["current"].iloc[slice_start:slice_end])

    return time_data, voltage, current, [slice_start, slice_end]

def fourier_series(x, f, n=0):
    """
    Returns a symbolic fourier series of order `n`.

    :param n: Order of the fourier series.
    :param x: Independent variable
    :param f: Frequency of the fourier series
    """
    # Make the parameter objects for all the terms
    a0, *cos_a = parameters(','.join(['a{}'.format(i) for i in range(0, n + 1)]))
    sin_b = parameters(','.join(['b{}'.format(i) for i in range(1, n + 1)]))
    # Construct the series
    series = a0 + sum(ai * cos(i * f * x) + bi * sin(i * f * x)
                     for i, (ai, bi) in enumerate(zip(cos_a, sin_b), start=1))
    return series

x, y = variables('x, y')
w, = parameters('w')
model_dict = {y: fourier_series(x, f=w, n=3)}
print(model_dict)

time_data, voltage, current, slice = load_data(36)
xdata = time_data
ydata = np.cos(9*xdata) + np.sin(9*xdata)

fit = Fit(model_dict, x = xdata, y = ydata)
fit_result = fit.execute()
print(fit_result)

fig, axs = plt.subplots()
axs.plot(xdata, ydata, label = "real", color = "blue")
axs.plot(xdata, fit.model(x = xdata, **fit_result.params).y, label = "fit", color = "red", linestyle = "dashed")
axs.legend()
fig.tight_layout()
plt.show()