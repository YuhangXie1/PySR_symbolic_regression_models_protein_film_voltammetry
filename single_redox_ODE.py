import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

#Reaction
"""
O + ne- --> R
"""

#Constants
F = 96485.3321          # s A / mol     #Faradays const
R = 8.3145              # J/ mol K      #Gas const
T = 298                 # K             #Temperature

#Variables
#n = 1                  #               #number of electrons transferred
A = 1e-4                # m^2           #Electrode area
Ru = 10                 # ohm           #Uncompensated resistance
Cdl = 1e-3              # As/V          #Double layer capacitance linear coefficient
gamma = 1e-6            # mol/m^2       #Surface coverage of the protein
k0 = 1e-4               # 1/s           #Standard rate constant
a = 0.3                 #               #Alpha = transfer coefficient, 0 <= a <= 1
E0 = 0                  # V             #Potential at equilibrium
frequency = 36          # V/s           #Frequency

#input voltage variables
w = 2*np.pi*36
p = np.pi/2
E_amplitude = 0.3
E_half = -0.05

#Non dimensionalisation
#constants
epsilon = (R*T)/F       #V
tau = epsilon/frequency #s
iota = (F*A*gamma)/tau  #A

#non-dims
Ru_nd = Ru*(iota/epsilon)
Cdl_nd = Cdl*(epsilon/(tau*iota))
k0_nd = k0*tau
E0_nd = E0/epsilon

#equations
def E_app(t):
    E_app = E_half + E_amplitude * np.sin(w * t + p)
    #Non dimensionalisation
    E_app = E_app/epsilon
    return E_app

def dE_app_dt(t):
    dE_app_dt = E_amplitude * w * np.cos(w * t + p)
    #Non dimensionalisation
    dE_app_dt = dE_app_dt*(tau/epsilon)
    return dE_app_dt

#ODE
def single_redox_ODE(t, y):
    #all expressions non-dimensionalised
    O = y[0]
    I_tot = y[1]

    E_eff = (E_app(t) - Ru_nd * I_tot)
    k_ox_nd = k0_nd * np.exp((1-a)*(E_eff-E0_nd))
    k_red_nd = k0_nd * np.exp((-a)*(E_eff-E0_nd))
    
    dO_dt = k_ox_nd*(1-O) - k_red_nd*O
    dI_dt = (1/Ru_nd)*dE_app_dt(t) + (1/(Ru_nd*Cdl_nd)) * dO_dt - (1/(Ru_nd*Cdl_nd)) * I_tot

    return [dO_dt, dI_dt]


O_0 = 0
I_tot_0 = 0
initial_y = [O_0,I_tot_0]
t_eval = np.linspace(0,1,100)
solution = solve_ivp(single_redox_ODE, [0,max(t_eval)], initial_y, t_eval=t_eval)


fig, axs = plt.subplots()
axs.plot(solution.t, solution.y[0], label = "[O]", color = "blue")
axs.plot(solution.t, solution.y[1], label = "I", color = "orange")
axs.plot(solution.t, E_app(solution.t), label = "Eapp", color = "green")
axs.legend()
plt.show()


