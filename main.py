from pysr import PySRRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

start_time = time.time()

""" data_volt = pd.read_csv("Data_for_eq_learning/9_Hz_2_cv_voltage", sep="\t", names=["time", "voltage"])
data_amp = pd.read_csv("Data_for_eq_learning/9_Hz_2_cv_current", sep="\t", names=["time","current"])

t_v = data_volt["time"]
y_v = data_volt["voltage"]

t_a = data_amp["time"]
y_a = data_amp["current"]

fig, axs = plt.subplots(3)
axs[0].plot(t_v,y_v)
axs[1].plot(t_a,y_a)
axs[2].plot(y_v,y_a)
plt.show() """

X = 2 * np.random.randn(100,5)
y = 2.5 * np.cos(X[:,3]) + X[:,0]**2 - 0.5

model = PySRRegressor(
    maxsize=20,
    niterations=40,
    binary_operators=["+","-","*","/","^"],
    unary_operators=[
        "cos",
        "sin",
        "tan",
        "asin",
        "acos",
        "atan",
        "sinh",
        "cosh",
        "tanh",
        "asinh",
        "acosh",
        "atanh",
        "exp",
        "log",
        "inv(x) = 1/x",
    ],
    extra_sympy_mappings={"inv":lambda x:1/x},
    elementwise_loss= "loss(prediction,target) = (prediction - target)^2"
)

model.fit(X,y)
print(model)
print(model.sympy())

end_time = time.time()
print(f"Time elapsed: {end_time - start_time}")