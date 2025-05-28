import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

#Reaction
"""
O + ne- --> R
"""

#Constants
F = 96485.3321  # s A / mol     #Faradays const
R = 8.3145      # J/ mol K      #Gas const
T = 298         # K             #Temperature
f = F/(R*T)     # s A / J       #Combination of F/RT constants

#Variables
n = 1           #               #number of electrons transferred
A = 1e-4        # m^2           #Electrode area
k0 = 1e-6       # 1/s           #Standard rate constant
a = 0.3         #               #Alpha = transfer coefficient, 0 <= a <= 1
E0 = 0          # V             #Potential at equilibrium
E_app = 1      # V             #Applied potential

kf = k0 * np.exp(-a*f*(E_app-E0)) #Forward reaction rate constant
kb = k0 * np.exp((1-a)*f*(E_app-E0)) #Backward reaction rate constant

def calc_current(O,R):
    i = n*F*A*(kf*O - kb*R)
    return i

#analytical solution
def analytical(t, initial):
    O_initial = initial[0]
    R_initial = initial[1]

    #solution in the form of X_ = c1 v1_ e^(l1*t) + c2 v2_ e^(l2*t)
    c1 = (O_initial + R_initial)/(kb + kf)
    c2 = O_initial - kb*c1
    
    O = c1*kb + c2*np.exp((-kf-kb)*t)
    R = c1*kf - c2*np.exp((-kf-kb)*t)

    return [O, R]

#ODE
def single_redox_ode(t, y):
    O, R = y
    #i = n*F*A*(kf*O - kb*R)
    #dO_dt = -i/(n*F*A)
    #dR_dt = i/(n*F*A)

    dO_dt = -(kf*O - kb*R)
    dR_dt = (kf*O - kb*R)

    return [dO_dt, dR_dt]


initial_y = [0,1]
t_eval = np.linspace(0,50,50)
solution = solve_ivp(single_redox_ode, [0,max(t_eval)], initial_y, t_eval=t_eval)
analytical_sol = analytical(t_eval, initial_y)

fig, axs = plt.subplots()
axs.plot(solution.t, solution.y[0], label = "[O]", color = "blue")
axs.plot(solution.t, solution.y[1], label = "[R]", color = "cyan")
axs.plot(t_eval, analytical_sol[0], label = "[O] analytical", color = "red", linestyle = "dotted")
axs.plot(t_eval, analytical_sol[1], label = "[R] analytical", color = "magenta", linestyle = "dotted")
#axs.plot(solution.t, calc_current(solution.y[0],solution.y[1]), label = "current", color = "green")
axs.legend()
plt.show()




