from pysr import PySRRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
from sympy import Symbol, sympify, expand, diff
from sympy.utilities.lambdify import lambdify

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
data_volt = pd.read_csv("Data_for_eq_learning/9_Hz_2_cv_voltage", sep="\t", names = ["time","voltage"])
data_amp = pd.read_csv("Data_for_eq_learning/9_Hz_2_cv_current", sep="\t", names = ["time","current"])
data_combined_9 = data_volt.join(data_amp["current"])
data_combined_9.insert(0, "Freq", 9)

#loading data
""" data_volt = pd.read_csv("Data_for_eq_learning/36_Hz_2_cv_voltage", sep="\t", names = ["time","voltage"])
data_amp = pd.read_csv("Data_for_eq_learning/36_Hz_2_cv_current", sep="\t", names = ["time","current"])
data_combined_36 = data_volt.join(data_amp["current"])
data_combined_36.insert(0, "Freq", 36) """

#combining data
""" data_combined = pd.concat([data_combined_9, data_combined_36]) """

#slicing data to remove start and end noise. Putting data into tuples
slice_start = 550
slice_end = -100
t_v = tuple(data_combined_9["time"].iloc[slice_start:slice_end])
y_v = tuple(data_combined_9["voltage"].iloc[slice_start:slice_end])
y_a = tuple(data_combined_9["current"].iloc[slice_start:slice_end])
freq = tuple(data_combined_9["Freq"].iloc[slice_start:slice_end])

y_a_pos = np.abs(y_a)


#shaping variables for pysr
X = np.array(t_v).reshape(-1,1)
y = np.array(y_v).reshape(-1,1)

""" freq_x = np.array(freq_pos).reshape(-1,1)
X = np.concatenate([X,freq_x],1) """


#pysr model define
model = PySRRegressor(
    maxsize=30,
    niterations=100,
    batching= True,
    binary_operators=["+","-","*","/","^"],
    unary_operators=[
        "cos",
        "sin",
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


#prediction
y_pred = np.cos((X + X)*(-28.322353))*(-0.3006429) - 1*0.049613677

x0 = Symbol("x0")
expression = sympify("cos((x0 + x0)*(-28.322353))*(-0.3006429) - 1*0.049613677")
expression_expanded = expand(expression)
print(expression_expanded)

func = lambdify(x0, expression_expanded, 'numpy')
print(func)

x0_array = np.array([0,1,2,3,4,5])
result_array = func(x0_array)
print(result_array)

differentiated = diff(expression_expanded, x0)
print(differentiated)

dv_dt = 17.0298286814874*np.sin(56.644706*np.array(t_v))

#bisecting line
y_bisect = 1.0335721e-5*dv_dt
y_reflect = []
dv_dt_trunc = []

for i in range(0,len(dv_dt)):
    if y_a[i] >= y_bisect[i]:
        calc = y_a[i]
        y_reflect.append(calc)
    else:
        calc = y_a[i] + (2*(y_bisect[i]-y_a[i]))
        y_reflect.append(calc)

#plotting
fig, axs = plt.subplots()
axs.plot(dv_dt, y_a, label = "data", color = "blue")
axs.plot(dv_dt, y_bisect, label = "bisect", color = "red")

#axs.plot(t_v, y_v, label = "data", color = "blue")
#axs.plot(t_v, y_pred, label = "pred", color = "red")
axs.plot(dv_dt, y_reflect, label = "reflected data", color = "orange")

#axs[1].plot(y_v,y_a, label = "pred", color = "red")
#axs.plot(dv_dt,y_a_pos, label = "pred", color = "orange")#
#axs.scatter(y_v_fit_pos, y_pred_multi_pos, label = "pos predict", color = "orange")
axs.set_xlabel("time")
axs.set_ylabel("voltage")

#axs.set_xlabel("dV/dt")
#axs.set_ylabel("current")
axs.legend(loc = "right", bbox_to_anchor = (1.25, 0.6))

fig.tight_layout()
plt.show()