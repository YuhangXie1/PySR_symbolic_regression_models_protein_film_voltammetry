from pysr import PySRRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

start_time = time.time()

""" data_volt = pd.read_csv("Data_for_eq_learning/9_Hz_2_cv_voltage", sep="\t", names = ["time","voltage"])
data_amp = pd.read_csv("Data_for_eq_learning/9_Hz_2_cv_current", sep="\t", names = ["time","current"])
 """
""" print (data_volt)
print (data_amp) """

""" slice_start = 1000
slice_end = 6000
t_v = tuple(data_volt["time"].iloc[slice_start:slice_end])
y_v = tuple(data_volt["voltage"].iloc[slice_start:slice_end])

t_a = tuple(data_amp["time"].iloc[slice_start:slice_end])
y_a = tuple(data_amp["current"].iloc[slice_start:slice_end]) """

"""
dv_dt = []
t_v_truncated = []
for i in range(0,len(y_v)):
    try:
        dv_dt.append((y_v[i+1] - y_v[i-1])/(t_v[i+1] - t_v[i-1]))
        t_v_truncated.append(t_v[i])
    except:
        pass """

#print(dv_dt)

#t_a = data_amp["time"]
#y_a = data_amp["current"]

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

X = np.linspace(-1,1,100)
y1 = np.sqrt(1-X**2)
y2 = -np.sqrt(1-X**2)

y = y1.reshape(-1,1)
X = X.reshape(-1,1)

model = PySRRegressor(
    maxsize=30,
    niterations=100,
    batching= True,
    binary_operators=["+","-","*","/","^"],
    unary_operators=[
        #"cos",
        #"sin",
        #"tan",
        #"asin",
        #"acos",
        #"atan",
        #"sinh",
        #"cosh",
        #"tanh",
        #"asinh",
        #"acosh",
        #"atanh",
        #"exp",
        #"log",
        #"inv",
        #"neg",
        "abs",
        "sign",
    ],
    elementwise_loss="loss(prediction, target) = (prediction - target)^2"
)

run = True
if run:
    model.fit(X,y)
    print(model)
    print(model.sympy())

end_time = time.time()
print(f"Time elapsed: {end_time - start_time}")



#y_pred = np.sin(np.acos(np.abs(X)))

fig, axs = plt.subplots(2)
#axs.plot(t_v_truncated, dv_dt)
axs[0].scatter(X, y)
#axs[1].plot(y_v, y_a)
#axs[0].plot(X, y_pred)
#axs.legend()
plt.show()