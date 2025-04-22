from pysr import PySRRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

start_time = time.time()

#loading data
data_volt = pd.read_csv("Data_for_eq_learning/9_Hz_2_cv_voltage", sep="\t", names = ["time","voltage"])
data_amp = pd.read_csv("Data_for_eq_learning/9_Hz_2_cv_current", sep="\t", names = ["time","current"])
data_combined = data_volt.join(data_amp["current"])

#slicing data to remove start and end noise. Putting data into tuples
slice_start = 200
slice_end = -100
t_v = tuple(data_combined["time"].iloc[slice_start:slice_end])
y_v = tuple(data_combined["voltage"].iloc[slice_start:slice_end])
y_a = tuple(data_combined["current"].iloc[slice_start:slice_end])

#calculating voltage derivative
dv_dt = []
t_v_truncated = []
y_a_truncated = []
for i in range(0,len(y_v)):
    try:
        dv_dt.append((y_v[i+1] - y_v[i-1])/(t_v[i+1] - t_v[i-1]))
        t_v_truncated.append(t_v[i])
        y_a_truncated.append(y_a[i])
    except:
        pass

#slicing data to positive and negative current
y_a_pos = []
y_v_fit_pos = []
y_a_neg = []
y_v_fit_neg = []
for i in range(0,len(y_a)):
    if y_a[i] >= 0:
        y_a_pos.append(y_a[i])
        y_v_fit_pos.append(y_v[i])
    else:
        y_a_neg.append(y_a[i])
        y_v_fit_neg.append(y_v[i])
        

""" X = np.array(y_v).reshape(-1,1)
y = np.array(y_a).reshape(-1,1) """

""" X = np.linspace(-1,1,100).reshape(-1,1)
y1 = np.sqrt(1-X**2)
y2 = -np.sqrt(1-X**2)
y = np.append(y1,y2)
X = np.append(X,X).reshape(-1,1)
y = y.reshape(-1,1)
print(X)
print(y) """

""" X = np.linspace(-1,1,100)
y1 = np.sqrt(1-X**2)
y2 = -np.sqrt(1-X**2)

y = y1.reshape(-1,1)
X = X.reshape(-1,1) """

#shaping variables for pysr
X = np.array(y_v_fit_neg).reshape(-1,1)
y = np.array(y_a_neg).reshape(-1,1)

#pysr model define
model = PySRRegressor(
    maxsize=30,
    niterations=100,
    batching= True,
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
        "inv",
        "neg",
        "abs",
        "sign",
    ],
    elementwise_loss="loss(prediction, target) = (prediction - target)^2"
)

run = False
if run:     
    model.fit(X,y)
    print(model)
    print(model.sympy())

end_time = time.time()
print(f"Time elapsed: {end_time - start_time}")


#equations trained from 1 arc
#y_pred = -0.00017439124*np.cos(np.exp(np.tanh(np.atan(np.atan(X)/np.sqrt(np.atan(X)**2 + 1))))/(np.sqrt(1 - np.tan(np.tan(np.atanh(np.mod(np.atanh(np.mod(np.tan(np.tan(np.tan(X))) + 1, 2) - 1) + 1, 2) - 1)))**2) - 0.6569488))
#y_pred_2 = np.cos(2.506182 / (np.cos(X) + -0.7301067)) * -0.0001696718

#equations trained from positive circle relationship
y_pred = X*(X*X*(X*(X + X) + X) - 0.0043396144)*(-0.025317585) - 1*(-0.00017110028)

#equations trained from negative circle relationship
y_pred = np.abs(X)*X*(-X - 0.18009685)*(-0.006479142) - 0.0001740445


fig, axs = plt.subplots(3)
axs[0].plot(t_v, y_v, label = "voltage")
axs[0].plot(t_v, y_a, label = "current")
#axs[0].plot(t_v_truncated,dv_dt, label = "dv/dt")
axs[0].legend()
#axs[0].scatter(X, y)
axs[1].scatter(y_v_fit_neg, y_a_neg)
axs[1].scatter(X,y_pred)
#axs[2].plot(dv_dt,y_a_truncated)
#axs[0].plot(X, y_pred)
plt.show()