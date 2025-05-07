import numpy as np

X = 2 * np.random.randn(100, 5)


time_start = 0
time_end = 100
time_step = 0.1

time_data = np.arange(time_start,time_end,time_step)
voltage = np.sin(time_data)
dv_dt = np.cos(time_data)


x = np.array(dv_dt)
a = np.array([1]*len(dv_dt))

X = np.array([a,x,x**2,x**3])
print (X)