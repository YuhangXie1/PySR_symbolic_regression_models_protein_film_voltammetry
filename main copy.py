from pysr import PySRRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
from sympy import Symbol, sympify, expand, diff
from sympy.utilities.lambdify import lambdify


def calculate_MSE(y, y_pred):

    mse = sum((y_pred - y)**2)
    print(f"MSE:{mse}")

    return mse

start_time = time.time()

loc=r".\Data_for_eq_learning"
files=os.listdir(loc)

""" data_complete = pd.DataFrame
for file in files[:1]:
    if "FTacV" in file:
        continue
    if "Hz" in file:
        split_file = file.split("_") 

        freq = split_file[:1]
        current_file = "_".join(freq+["Hz","2", "cv","current"])
        potential_file = "_".join(freq+["Hz","2", "cv","voltage"])
        
        data_potential = pd.read_csv(os.path.join(loc, potential_file), sep="\t", names = ["time","voltage"])
        data_current = pd.read_csv(os.path.join(loc, current_file), sep="\t", names = ["time","current"])
        data_combined = data_potential.join(data_current["current"])
        data_combined.insert(0, "freq", freq) """


#loading data
data_volt_2 = pd.read_csv("Data_for_eq_learning/36_Hz_2_cv_voltage", sep="\t", names = ["time","voltage"])
data_amp_2 = pd.read_csv("Data_for_eq_learning/36_Hz_2_cv_current", sep="\t", names = ["time","current"])
data_combined_9_2 = data_volt_2.join(data_amp_2["current"])
data_combined_9_2.insert(0, "Freq", 36)

#loading data
""" data_volt = pd.read_csv("Data_for_eq_learning/36_Hz_2_cv_voltage", sep="\t", names = ["time","voltage"])
data_amp = pd.read_csv("Data_for_eq_learning/36_Hz_2_cv_current", sep="\t", names = ["time","current"])
data_combined_36 = data_volt.join(data_amp["current"])
data_combined_36.insert(0, "Freq", 36) """
data_volt_1 = pd.read_csv("Data_for_eq_learning/9_Hz_1_cv_voltage", sep="\t", names = ["time","voltage"])
data_amp_1 = pd.read_csv("Data_for_eq_learning/9_Hz_1_cv_current", sep="\t", names = ["time","current"])
data_combined_9_1 = data_volt_1.join(data_amp_1["current"])
data_combined_9_1.insert(0, "Freq", 9)

#combining data
""" data_combined = pd.concat([data_combined_9, data_combined_36]) """

#slicing data to remove start and end noise. Putting data into tuples
#slice_start = 550
#slice_end = -100
slice_start = 550
slice_end = -100
t_v_2 = tuple(data_combined_9_2["time"].iloc[slice_start:slice_end])
y_v_2 = tuple(data_combined_9_2["voltage"].iloc[slice_start:slice_end])
y_a_2 = tuple(data_combined_9_2["current"].iloc[slice_start:slice_end])

t_v_1 = tuple(data_combined_9_1["time"].iloc[slice_start:slice_end])
y_v_1 = tuple(data_combined_9_1["voltage"].iloc[slice_start:slice_end])
y_a_1 = tuple(data_combined_9_1["current"].iloc[slice_start:slice_end])


#shaping variables for pysr
X = np.array(t_v_2).reshape(-1,1)
y = np.array(y_v_2).reshape(-1,1)

""" freq_x = np.array(freq_pos).reshape(-1,1)
X = np.concatenate([X,freq_x],1) """


#pysr model define
model = PySRRegressor(
    maxsize=30,
    niterations=100,
    batching= True,
    binary_operators=["+","*","^"],
    unary_operators=["sin","cos"],
    #nested_constraints={"sin":{"sin":0}},
    elementwise_loss="loss(prediction, target) = (prediction - target)^2"
)

run = False
if run:     
    model.fit(X,y)
    print(model)
    print(model.sympy())
    
#voltage_pred = model.predict(X)
end_time = time.time()
print(f"Time elapsed: {end_time - start_time}")


#calculating analytical values
t_v_2 = np.array(t_v_2)
#voltage_fit_2 = -0.3006429*np.cos(56.644706*t_v_2) - 0.049613677
voltage_fit_2 = 0.000165404485499355*np.sin(0.893943594923073*t_v_2 + 1.38349668241203) - 0.0495928256240627
dv_dt_2 = 17.0298286814874*np.sin(56.644706*t_v_2)

""" t_v_1 = np.array(t_v_1)
voltage_fit_1 = 0.3005573*np.sin(56.64424*t_v_1 + 42.415722374584) - 0.049560416
dv_dt_1 = 17.024839834952*np.cos(56.64424*t_v_1 + 42.415722374584) """

t_v_1 = np.array(t_v_1)
voltage_fit_1 = -0.3006429*np.cos(56.644706*t_v_1) - 0.049613677
dv_dt_1 = 17.0298286814874*np.sin(56.644706*t_v_1)


""" x0 = Symbol("x0")
expression = sympify("sin((x0 + 0.7488091)*(-56.64424))*(-0.3005573) - 0.049560416")
expression_expanded = expand(expression)
print(expression_expanded)

differentiated = diff(expression_expanded, x0)
print(differentiated) """

""" x0 = np.array(y_v_2)
x1 = x0**2
x2 = x0**3
x3 = dv_dt_2
x4 = dv_dt_2**2
x5 = dv_dt_2**3
current_pred_2 = 5.617385e-5*x0**2 + 0.0001123477*x0*x1 + 4.7648637e-6*x0*x3 + 0.00012237591814345*x0 + 5.617385e-5*x1**2 + 4.7648637e-6*x1*x3 + 0.00012237591814345*x1 + 1.04536201768389e-5*x3 - 1.89491896944645e-6

x0 = np.array(y_v_1)
x1 = x0**2
x2 = x0**3
x3 = dv_dt_1
x4 = dv_dt_1**2
x5 = dv_dt_1**3
current_pred_1 = 5.617385e-5*x0**2 + 0.0001123477*x0*x1 + 4.7648637e-6*x0*x3 + 0.00012237591814345*x0 + 5.617385e-5*x1**2 + 4.7648637e-6*x1*x3 + 0.00012237591814345*x1 + 1.04536201768389e-5*x3 - 1.89491896944645e-6
 """

x = np.array(y_v_2)
dx = dv_dt_2
current_pred_2 = 4.7648637e-6*dx*x**2 + 4.7648637e-6*dx*x + 1.04536201768389e-5*dx + 5.617385e-5*x**4 + 0.0001123477*x**3 + 0.00017854976814345*x**2 + 0.00012237591814345*x - 1.89491896944645e-6
MSE_2 = calculate_MSE(y_a_2,current_pred_2)

x = np.array(y_v_1)
dx = dv_dt_1
current_pred_1 = 4.7648637e-6*dx*x**2 + 4.7648637e-6*dx*x + 1.04536201768389e-5*dx + 5.617385e-5*x**4 + 0.0001123477*x**3 + 0.00017854976814345*x**2 + 0.00012237591814345*x - 1.89491896944645e-6
MSE_1 = calculate_MSE(y_a_1,current_pred_1)



fig, axs = plt.subplots(2)

slice_start = 0
slice_end = -1

axs[0].plot(t_v_2[slice_start:slice_end], y_v_2[slice_start:slice_end], label = "data", color = "cyan")
axs[0].plot(t_v_2[slice_start:slice_end], voltage_fit_2[slice_start:slice_end], label = "pred", color = "red", linestyle = "dotted")
axs[0].set_xlabel("time")
axs[0].set_ylabel("current")
axs[0].set_title(f"File 2. MSE = {MSE_2}")
axs[0].legend(loc = "right", bbox_to_anchor = (1.25, 0.6))

axs[1].plot(t_v_1[slice_start:slice_end], y_a_1[slice_start:slice_end], label = "data", color = "orange")
axs[1].plot(t_v_1[slice_start:slice_end], current_pred_1[slice_start:slice_end], label = "pred", color = "red", linestyle = "dotted")
axs[1].set_xlabel("time")
axs[1].set_ylabel("current")
axs[1].set_title(f"File 1. MSE = {MSE_1}")
axs[1].legend(loc = "right", bbox_to_anchor = (1.25, 0.6))


""" axs[0].plot(t_v_2[slice_start:slice_end], y_v_2[slice_start:slice_end], label = "data", color = "cyan")
axs[0].plot(t_v_2[slice_start:slice_end], voltage_fit_2[slice_start:slice_end], label = "pred", color = "red", linestyle = "dotted")
axs[0].set_xlabel("time")
axs[0].set_ylabel("voltage")
axs[0].set_title("File 2")
axs[0].legend(loc = "right", bbox_to_anchor = (1.25, 0.6))

axs[1].plot(t_v_1[slice_start:slice_end], y_v_1[slice_start:slice_end], label = "data", color = "orange")
axs[1].plot(t_v_1[slice_start:slice_end], voltage_fit_1[slice_start:slice_end], label = "pred", color = "red", linestyle = "dotted")
axs[1].set_xlabel("time")
axs[1].set_ylabel("voltage")
axs[1].set_title("File 1")
axs[1].legend(loc = "right", bbox_to_anchor = (1.25, 0.6)) """

fig.tight_layout()
plt.show()

fig, axs = plt.subplots(2)

slice_start = 0
slice_end = -1

axs[0].plot(dv_dt_2[slice_start:slice_end], y_a_2[slice_start:slice_end], label = "data", color = "cyan")
axs[0].plot(dv_dt_2[slice_start:slice_end], current_pred_2[slice_start:slice_end], label = "pred", color = "red", linestyle = "dotted")
axs[0].set_xlabel("dv/dt")
axs[0].set_ylabel("current")
axs[0].set_title("File 2")
axs[0].legend(loc = "right", bbox_to_anchor = (1.25, 0.6))

axs[1].plot(dv_dt_1[slice_start:slice_end], y_a_1[slice_start:slice_end], label = "data", color = "orange")
axs[1].plot(dv_dt_1[slice_start:slice_end], current_pred_1[slice_start:slice_end], label = "pred", color = "red", linestyle = "dotted")
axs[1].set_xlabel("dv/dt")
axs[1].set_ylabel("current")
axs[1].set_title("File 1")
axs[1].legend(loc = "right", bbox_to_anchor = (1.25, 0.6))

fig.tight_layout()
plt.show()