from pysr import PySRRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

start_time = time.time()

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
""" data_combined = pd.concat([data_combined_9, data_combined_36])
 """
#slicing data to remove start and end noise. Putting data into tuples
slice_start = 200
slice_end = -100
t_v = tuple(data_combined_9["time"].iloc[slice_start:slice_end])
y_v = tuple(data_combined_9["voltage"].iloc[slice_start:slice_end])
y_a = tuple(data_combined_9["current"].iloc[slice_start:slice_end])
freq = tuple(data_combined_9["Freq"].iloc[slice_start:slice_end])

#calculating voltage derivative
dv_dt = []
t_v_truncated = []
y_a_truncated = []
y_v_truncated = []
freq_truncated = []
for i in range(0,len(y_v)):
    try:
        dv_dt.append((y_v[i+1] - y_v[i-1])/(t_v[i+1] - t_v[i-1]))
        t_v_truncated.append(t_v[i])
        y_a_truncated.append(y_a[i])
        y_v_truncated.append(y_v[i])
        freq_truncated.append(freq[i])
    except:
        pass

#slicing data to positive and negative current
y_a_pos = []
y_v_fit_pos = []
freq_pos = []
y_a_neg = []
y_v_fit_neg = []
freq_neg = []
for i in range(0,len(y_a)):
    if y_a[i] >= 0:
        y_a_pos.append(y_a[i])
        y_v_fit_pos.append(y_v[i])
        freq_pos.append(freq[i])
    else:
        y_a_neg.append(y_a[i])
        y_v_fit_neg.append(y_v[i])
        freq_neg.append(freq[i])
        

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
""" X = np.array(y_v_fit_neg).reshape(-1,1)
y = np.array(y_a_neg).reshape(-1,1) """

X = np.array(dv_dt).reshape(-1,1)
y = np.array(y_a_truncated).reshape(-1,1)

freq_x = np.array(freq_truncated).reshape(-1,1)
X = np.concatenate([X,freq_x],1)


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
x_neg = np.array(y_v_fit_neg).reshape(-1,1)
x_pos = np.array(y_v_fit_pos).reshape(-1,1)
f = 9

y_pred_pos = x_pos*(x_pos*x_pos*(x_pos*(x_pos + x_pos) + x_pos) - 0.0043396144)*(-0.025317585) - 1*(-0.00017110028)

#equations trained from negative circle relationship
#y_pred_neg = np.abs(x)*x*(-x - 0.18009685)*(-0.006479142) - 0.0001740445
y_pred_neg = -0.00029978607 - 0.0001190309/np.cos((x_neg - f)*3.8130171)

#prediction on dv_dt
#y_pred = x*1.0125009e-5

fig, axs = plt.subplots()
""" axs[0].plot(data_combined_9["voltage"], data_combined_9["current"], label = "9 Hz")
axs[0].plot(data_combined_36["voltage"], data_combined_36["current"], label = "36 Hz") """
""" axs[0].plot(t_v_truncated,dv_dt, label = "dv/dt")
axs[0].set_xlabel("time")
axs[0].set_ylabel("dV/dt") """
#axs[0].legend()
#axs[0].scatter(X, y)
""" axs[1].scatter(y_v_fit_neg, y_a_neg, marker = ".")
axs[1].scatter(x,y_pred_neg, marker = ".") """
axs.scatter(y_v_fit_pos, y_pred_pos, label = "postive current prediction", color = "red")
axs.scatter(y_v_fit_neg, y_pred_neg, label = "negative current prediction", color = "orange")
axs.plot(y_v_truncated, y_a_truncated, label = "data", color = "blue")
axs.set_xlabel("voltage")
axs.set_ylabel("current")
axs.legend(loc = "center")
#axs[2].plot(dv_dt,y_a_truncated)
#axs[0].plot(X, y_pred)
fig.tight_layout()
plt.show()