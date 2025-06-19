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
#n = 1           #               #number of electrons transferred
A = 1e-4        # m^2           #Electrode area
Ru = 100         # ohm           #Uncompensated resistance
Cdl = 1e-3         #               #Double layer capacitance linear coefficient
gamma = 1e-4       # mol/m^2       #Surface coverage of the protein
k0 = 1e-6       # 1/s           #Standard rate constant
a = 0.3         #               #Alpha = transfer coefficient, 0 <= a <= 1
E0 = 2          # V             #Potential at equilibrium

#kf = k0 * np.exp(-a*f*(E_app-E0)) #Forward reaction rate constant
#kb = k0 * np.exp((1-a)*f*(E_app-E0)) #Backward reaction rate constant

kf = k0 * (1/3) #Forward reaction rate constant
kb = k0 - kf #Backward reaction rate constant

#variables
w = 2*np.pi*36
p = np.pi/2
E_amplitude = 0.3
E_half = 0

#Non dimensionalisation
kf = kf/k0
kb = kb/k0
Ru = ((F**2 * k0)/(R*T))*Ru
Cdl = ((R*T)/F**2)*Cdl

#equations
def E_app(t):
    E_app = E_half + E_amplitude * np.sin(w * t + p)
    #Non dimensionalisation
    E_app = (F/(R*T))*(E_app - E0)
    return E_app

def dE_app_dt(t):
    dE_app_dt = E_amplitude * w * np.cos(w * t + p)
    #Non dimensionalisation
    dE_app_dt = (F/(R*T))*dE_app_dt
    return dE_app_dt

#ODE
def single_redox_ODE(t, y):
    O = y[0]
    I_tot = y[1]

    E_eff = E_app(t) - Ru * I_tot
    dO_dt = E_eff*(kb*(1-O) - kf*O)

    dI_dt = (1/(Ru*Cdl))*(Cdl*dE_app_dt(t) + F*A*gamma* dO_dt - I_tot)

    return [dO_dt, dI_dt]


O_0 = 0
I_tot_0 = 0
initial_y = [O_0,I_tot_0]
t_eval = np.linspace(0,1,100)
solution = solve_ivp(single_redox_ODE, [0,max(t_eval)], initial_y, t_eval=t_eval)


fig, axs = plt.subplots()
axs.plot(solution.t, solution.y[0], label = "[O]", color = "blue")
axs.plot(solution.t, solution.y[1], label = "I", color = "orange")
axs.legend()
plt.show()


