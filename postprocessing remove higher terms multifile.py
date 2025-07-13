import os
import numpy as np
import pandas as pd
import sympy as sp
from sympy import Symbol, symbols, sympify, diff, collect, expand
from sympy.utilities import lambdify
from scipy.optimize import curve_fit
from itertools import product
import matplotlib.pyplot as plt
from pathlib import Path
import csv


def cos_similarity(x,y):
    return np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))

def calculate_MSE(y, y_pred):
    """Returns the mean squared error between a y value (real) and a predicted y value (prediction)."""
    mse = sum((y_pred - y)**2)/len(y)
    return mse

def fourier_transform(current, time_data):
   
    ft=np.fft.fft(current)
    freqs=np.fft.fftfreq(len(time_data), time_data[1]-time_data[0])

    return freqs, ft
    
def filter_fft(freqs, ft, band_size, Hz, desired_harmonic):
    freq_loc=desired_harmonic*int(Hz)
    filtered_ft=np.zeros(len(ft), dtype="complex")
    for sign in [-1, 1]:
        ft_loc=np.where((freqs>sign*freq_loc-(band_size*int(Hz))) & (freqs<sign*freq_loc+(band_size*int(Hz))))
        filtered_ft[ft_loc]=ft[ft_loc]
    
    return filtered_ft

def fit_sin(tt, yy):
    """
    Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"
    Obtained from https://stackoverflow.com/questions/16716302/how-do-i-fit-a-sine-curve-to-my-data-with-pylab-and-numpy, user unsym, Feb 19, 2017

    """
    tt = np.array(tt)
    yy = np.array(yy)
    ff = np.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing
    Fyy = abs(np.fft.fft(yy))
    guess_freq = abs(ff[np.argmax(Fyy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
    guess_amp = np.std(yy) * 2.**0.5
    guess_offset = np.mean(yy)
    guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset])

    def sinfunc(t, A, w, p, c):  return A * np.sin(w*t + p) + c
    popt, pcov = curve_fit(sinfunc, tt, yy, p0=guess)
    A, w, p, c = popt
    f = w/(2.*np.pi)
    fitfunc = lambda t: A * np.sin(w*t + p) + c
    output_dict = {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1./f, "fitfunc": fitfunc, "maxcov": np.max(pcov), "rawres": (guess,popt,pcov)}

    #lambdifying
    x0 = Symbol("x0")
    voltage_equation = A*sp.sin(w*x0 + p) + c

    return voltage_equation, [A,w,p,c]

def fit_voltage_eqn(time_data, voltage):
    voltage_eqn, coeff_array = fit_sin(time_data, voltage)
    
    x0 = Symbol("x0")
    dv_dt_eqn = diff(voltage_eqn, x0)

    return voltage_eqn, dv_dt_eqn, coeff_array

def load_data(Hz):
    """
    Returns numpy arrays of time_data, voltage, current and frequency for a set of data files
    """
    #loading data
    data_volt = pd.read_csv(f"Data_for_eq_learning/{Hz}_Hz_2_cv_voltage", sep="\t", names = ["time","voltage"])
    data_amp = pd.read_csv(f"Data_for_eq_learning/{Hz}_Hz_2_cv_current", sep="\t", names = ["time","current"])
    
    #data_amp = pd.read_csv(r"Data_for_eq_learning\FTacV_before_PSV_cv_current", sep="\t", names = ["time","current"])
    #data_volt = pd.read_csv(r"Data_for_eq_learning\FTacV_before_PSV_cv_voltage", sep="\t", names = ["time","voltage"])

    #slicing data to remove start and end noise. Putting data into tuples
    slice_start = 550
    slice_end = -100
    time_data = np.array(data_amp["time"].iloc[slice_start:slice_end])
    voltage = np.array(data_volt["voltage"].iloc[slice_start:slice_end])
    current = np.array(data_amp["current"].iloc[slice_start:slice_end])

    return time_data, voltage, current, [slice_start, slice_end]

def load_data_2(filepath):
    data = pd.read_csv(filepath, header=0)

    slice_start = 550
    slice_end = -100
    time_data = np.array(data["x"].iloc[slice_start:slice_end])
    voltage = np.array(data["z"].iloc[slice_start:slice_end])
    current = np.array(data["y"].iloc[slice_start:slice_end])

    return time_data, voltage, current, [slice_start, slice_end]

def generate_plots(time_data, current, dv_dt, current_pred, eqn_number, terms, output_filepath):

    fig, axs = plt.subplots(3)
    axs[0].plot(time_data, current, label = "data", color = "blue")
    axs[0].plot(time_data, current_pred, label = "prediction", color = "red")
    axs[0].set_xlabel("time")
    axs[0].set_ylabel("current")
    axs[0].set_title(f"Eqn {eqn_number}. Terms removed: {terms}. Current vs time. Whole. ")
    axs[0].legend()

    slice_start = 10550
    slice_end = 20000
    axs[1].plot(time_data[slice_start:slice_end], current[slice_start:slice_end], label = "data", color = "blue")
    axs[1].plot(time_data[slice_start:slice_end], current_pred[slice_start:slice_end], label = "pred", color = "red", linestyle = "dotted")
    axs[1].set_xlabel("time")
    axs[1].set_ylabel("current")
    axs[1].set_title(f"Eqn {eqn_number}. Terms removed: {terms}. Current vs time. Slice {slice_start}-{slice_end}.")
    axs[1].legend()

    slice_start = 30550
    slice_end = 40000
    axs[2].plot(time_data[slice_start:slice_end], current[slice_start:slice_end], label = "data", color = "blue")
    axs[2].plot(time_data[slice_start:slice_end], current_pred[slice_start:slice_end], label = "pred", color = "red", linestyle = "dotted")
    axs[2].set_xlabel("time")
    axs[2].set_ylabel("current")
    axs[2].set_title(f"Eqn {eqn_number}. Terms removed: {terms}. Current vs time. Slice {slice_start}-{slice_end}.")
    axs[2].legend()

    fig.tight_layout()
    plt.savefig(os.path.join(output_filepath, f"eqn_{eqn_number}",f"removed_{terms}_terms", f"Eqn-{eqn_number}-rm-{terms}-current-time.png"))
    plt.close(fig.figure)

    #current dV/dt graph
    fig, axs = plt.subplots()
    axs.plot(dv_dt, current, label = "data", color = "blue")
    axs.plot(dv_dt, current_pred, label = "prediction", color = "red")
    axs.set_xlabel("dV/dt")
    axs.set_ylabel("current")
    axs.set_title(f"Eqn {eqn_number}. Terms removed: {terms}. Current vs dV/dt.")
    axs.legend()

    fig.tight_layout()
    plt.savefig(os.path.join(output_filepath, f"eqn_{eqn_number}",f"removed_{terms}_terms", f"Eqn-{eqn_number}-rm-{terms}-dv_dt.png"))
    plt.close(fig.figure)
    
def generate_fft_graphs(time_data, current, current_pred, eqn_number, terms, output_filepath):
    band_size=0.1

    #calculating
    freqs, ft = fourier_transform(current, time_data)
    freqs_pred, ft_pred = fourier_transform(current_pred, time_data)

    #ploting all harmonics
    fig, axs = plt.subplots()
    axs.plot(freqs, np.log10(ft**2), color = "blue", label="real")
    axs.plot(freqs_pred, np.log10(ft_pred**2), color = "red", label="pred")
    axs.set_xlim(0,20*Hz)
    axs.set_ylim(0, 12)
    axs.set_xlabel("Frequency")
    axs.set_ylabel("Log10 ft^2")
    axs.set_title(f"File {Hz}Hz, Eqn {eqn_number}, removed {terms} terms, all harmonics")
    axs.legend()
    fig.tight_layout()
    plt.savefig(os.path.join(output_filepath, f"eqn_{eqn_number}",f"removed_{terms}_terms", "All-harmonic-freq.png"))

    #plotting figures 1 by 1
    for harmonic in range(1,10):
        
        #plotting all freq domain graphs
        fig, axs = plt.subplots()

        ft_filtered = filter_fft(freqs, ft, band_size, Hz, harmonic)
        ft_filtered_pred = filter_fft(freqs_pred, ft_pred, band_size, Hz, harmonic)
        
        log_ft_filtered = np.log10(ft_filtered**2)
        log_ft_filtered_pred = np.log10(ft_filtered_pred**2)
        
        axs.plot(freqs, log_ft_filtered, color = "blue", label="real")
        axs.plot(freqs_pred, log_ft_filtered_pred, color = "red", label="pred")
        axs.set_xlim(harmonic*int(Hz)-harmonic*int(Hz)*0.03, harmonic*int(Hz)+harmonic*int(Hz)*0.03)
        axs.legend()
        axs.set_xlabel("Frequency (Hz)")
        axs.set_ylabel("Log10 ft^2")
        axs.set_title(f"File {Hz}Hz, Eqn {eqn_number}, removed {terms} terms. Harmonic {harmonic} in freq domain")
        fig.tight_layout()
        Path(os.path.join(output_filepath, f"eqn_{eqn_number}",f"removed_{terms}_terms","freq_domain")).mkdir(parents=True, exist_ok=True)
        plt.savefig(os.path.join(output_filepath, f"eqn_{eqn_number}",f"removed_{terms}_terms","freq_domain", f"Harmonic-{harmonic}-freq.png"))
        plt.close(fig.figure)

        inverseft=np.fft.ifft(ft_filtered)
        inverseft_pred=np.fft.ifft(ft_filtered_pred)

        #plotting all time domain graphs
        fig, axs = plt.subplots()
        axs.plot(time_data, inverseft, color = "blue", label="real")
        axs.plot(time_data, inverseft_pred, color = "red", label="pred")
        axs.legend()
        axs.set_xlabel("Time")
        axs.set_ylabel("Current")
        axs.set_title(f"File {Hz}Hz, Eqn {eqn_number}, removed {terms} terms. Harmonic {harmonic} in time domain")
        fig.tight_layout()
        Path(os.path.join(output_filepath, f"eqn_{eqn_number}",f"removed_{terms}_terms","time_domain")).mkdir(parents=True, exist_ok=True)
        plt.savefig(os.path.join(output_filepath, f"eqn_{eqn_number}",f"removed_{terms}_terms", "time_domain", f"Eqn-{eqn_number}-harmonic-{harmonic}-freq.png"))
        plt.close(fig.figure)


        #plotting all voltage domain graphs
        fig, axs = plt.subplots()
        axs.plot(voltage, inverseft, color = "blue", label="real")
        axs.plot(voltage, inverseft_pred, color = "red", label="pred")
        axs.legend()
        axs.set_xlabel("Voltage")
        axs.set_ylabel("Current")
        axs.set_title(f"File {Hz}Hz, Eqn {eqn_number}, removed {terms} terms. Harmonic {harmonic} in voltage domain")
        fig.tight_layout()
        Path(os.path.join(output_filepath, f"eqn_{eqn_number}",f"removed_{terms}_terms","voltage_domain")).mkdir(parents=True, exist_ok=True)
        plt.savefig(os.path.join(output_filepath, f"eqn_{eqn_number}",f"removed_{terms}_terms", "voltage_domain", f"Eqn-{eqn_number}-harmonic-{harmonic}-freq.png"))
        plt.close(fig.figure)

    #Plotting 3 by 3 figures

    fig1, axs1 = plt.subplots(3,3)
    axs1 = axs1.flatten()

    fig2, axs2 = plt.subplots(3,3)
    axs2 = axs2.flatten()

    fig3, axs3 = plt.subplots(3,3)
    axs3 = axs3.flatten()

    for harmonic in range(1,10):
        ax1 = axs1[harmonic - 1]

        ft_filtered = filter_fft(freqs, ft, band_size, Hz, harmonic)
        ft_filtered_pred = filter_fft(freqs_pred, ft_pred, band_size, Hz, harmonic)
        
        log_ft_filtered = np.log10(ft_filtered**2)
        log_ft_filtered_pred = np.log10(ft_filtered_pred**2)
        
        ax1.plot(freqs, log_ft_filtered, color = "blue", label="real")
        ax1.plot(freqs_pred, log_ft_filtered_pred, color = "red", label="pred")
        ax1.set_xlim(harmonic*int(Hz)-harmonic*int(Hz)*0.03, harmonic*int(Hz)+harmonic*int(Hz)*0.03)
        ax1.set_title(f"Harmonic {harmonic}")

        handles_1, labels_1 = ax1.get_legend_handles_labels()

        ax2 = axs2[harmonic - 1]

        inverseft=np.fft.ifft(ft_filtered)
        inverseft_pred=np.fft.ifft(ft_filtered_pred)

        ax2.plot(time_data, inverseft, color = "blue", label="real")
        ax2.plot(time_data, inverseft_pred, color = "red", label="pred")
        ax2.set_title(f"Harmonic {harmonic}")

        handles_2, labels_2 = ax2.get_legend_handles_labels()

        ax3 = axs3[harmonic - 1]
        ax3.plot(voltage, inverseft, color = "blue", label="real")
        ax3.plot(voltage, inverseft_pred, color = "red", label="pred")
        ax3.set_title(f"Harmonic {harmonic}")

        handles_3, labels_3 = ax3.get_legend_handles_labels()

    fig1.legend(handles_1, labels_1, loc='lower right')
    fig1.supxlabel("Frequency (Hz)")
    fig1.supylabel("Log10 ft^2")
    fig1.suptitle(f"File {Hz}Hz, Eqn {eqn_number}, removed {terms} terms. Harmonics in freq domain")
    fig1.tight_layout()
    fig1.savefig(os.path.join(output_filepath, f"eqn_{eqn_number}",f"removed_{terms}_terms", f"Harmonics-freq-3x3.png"))
    plt.close(fig1.figure)

    fig2.legend(handles_2, labels_2, loc='lower right')
    fig2.supxlabel("Time")
    fig2.supylabel("Current")
    fig2.suptitle(f"File {Hz}Hz, Eqn {eqn_number}, removed {terms} terms. Harmonics in time domain")
    fig2.tight_layout()
    fig2.savefig(os.path.join(output_filepath, f"eqn_{eqn_number}",f"removed_{terms}_terms", f"Harmonics-time-3x3.png"))
    plt.close(fig2.figure)

    fig3.legend(handles_3, labels_3, loc='lower right')
    fig3.supxlabel("Voltage")
    fig3.supylabel("Current")
    fig3.suptitle(f"File {Hz}Hz, Eqn {eqn_number}, removed {terms} terms. Harmonics in voltage domain")
    fig3.tight_layout()
    fig3.savefig(os.path.join(output_filepath, f"eqn_{eqn_number}",f"removed_{terms}_terms", f"Harmonics-voltage-3x3.png"))
    plt.close(fig3.figure)

def combine_data(files_freq):
    #find voltage eqn and stitch data together
    combined_data_array = np.empty((1,10))
    for Hz in files_freq:
        
        #load data
        time_data, voltage, current, slice_array = load_data(Hz)
        voltage_eqn, dv_dt_eqn, coeff_array = fit_voltage_eqn(time_data, voltage)

        #current phase and amplitude data
        phase_data = pd.read_csv(rf"results\20250617-current-phase-original-set\summary.csv")
        p_val = np.array(phase_data.loc[phase_data["Hz"] == Hz]["phase"])[0]
        A_val = np.array(phase_data.loc[phase_data["Hz"] == Hz]["amplitude"])[0]
        w = coeff_array[1]

        #time shift t
        time_data = time_data + (p_val/w)

        #new time shifted voltage and dv_dt
        x0 = Symbol("x0")
        voltage_lambda = lambdify(x0,voltage_eqn)
        voltage = voltage_lambda(time_data)
        dv_dt_lambda = lambdify(x0,dv_dt_eqn)
        dv_dt = dv_dt_lambda(time_data)/w #normalise dV/dt

        #normalise current
        current = current/A_val

        A = np.full((1,len(time_data)), coeff_array[0])[0]
        w = np.full((1,len(time_data)), coeff_array[1])[0]
        p = np.full((1,len(time_data)), coeff_array[2])[0]
        c = np.full((1,len(time_data)), coeff_array[3])[0]
        p_val = np.full((1,len(time_data)), p_val)[0]
        A_val = np.full((1,len(time_data)), A_val)[0]

        combined_data_array = np.concatenate([combined_data_array, np.array([time_data, voltage, dv_dt, current, A, w, p, c, A_val, p_val]).T],0)
    combined_data_array = np.delete(combined_data_array, (0), axis=0)
    return combined_data_array

def get_coefficient_and_terms(current_eqn_sympy, max_power):
    coefficient_dict = {}
    power_terms_dict = {}
    for power_x in np.arange(0,max_power+1,1):
        for power_dx in np.arange(0, max_power+1,1):
            coefficient_dict[x**power_x * dx**power_dx] = current_eqn_sympy.coeff(x,power_x).coeff(dx,power_dx)
            if current_eqn_sympy.coeff(x,power_x).coeff(dx,power_dx) != 0:
                power_terms_dict[x**power_x * dx**power_dx] = int(power_x + power_dx)

    return coefficient_dict, power_terms_dict


##main##
#files_freq = [9, 36, 45, 54, 63, 72, 81, 90, 99]
files_freq = [9]
Hz = files_freq[0]

load_filepath = rf"results\20250620-multi-fit-workflow-9\time-shift-normalised_dv_dt_aval"
output_filepath = rf"results/20250624-normalised-no-div/fft-remove-higher-terms-eqn-0-FILE1-{Hz}Hz"
Path(output_filepath).mkdir(parents=True, exist_ok=True)

combined_data_array = combine_data(files_freq)
time_data = combined_data_array[:,0]
voltage = combined_data_array[:,1]
dv_dt = combined_data_array[:,2]
current = combined_data_array[:,3]
A_dat = combined_data_array[:,4]
w_dat = combined_data_array[:,5]
p_dat = combined_data_array[:,6]
c_dat = combined_data_array[:,7]
A_val_dat = combined_data_array[:,8]
p_val_dat = combined_data_array[:,9]

""" current_summary_file = pd.read_csv(os.path.join(load_filepath, "summary_current_model.csv"), header=0)
data_equations = sympify(np.array(current_summary_file["substituted_form"]))
current_eqn = [data_equations[0]] """

current_eqn = "1.1937383*dx*p_val*x**2 + 2.3874766*dx*p_val*x + 1.035643*dx*x**2 + 2.071286*dx*x + 3.3564444*dx - 0.100718481781239*p*p_val*x**2 - 0.08737961295819*p*x**2 - 0.24831903*p_val*x + 0.51501876*x**2 - 0.017753009"

x, dx, A, w, p, c, A_val, p_val = symbols("x dx A w p c A_val p_val")
current_eqn_sympy = sympify(current_eqn)
coefficient_dict, power_terms_dict = get_coefficient_and_terms(current_eqn_sympy, 4)

data_equations = [current_eqn_sympy]
""" fig_overall, axs_overall = plt.subplots()
axs_overall.set_xlabel("Highest total power of variables")
axs_overall.set_ylabel("Log MSE")
axs_overall.set_title("All equations, highest power vs MSE")

overall_powers = []
overall_log_MSE = [] """

for eqn_number, eqn in enumerate(data_equations):

    Path(os.path.join(output_filepath, f"eqn_{eqn_number}")).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(output_filepath, f"eqn_{eqn_number}", f"summary_eqn_{eqn_number}.csv"), "a", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["num_terms_removed",
                        "MSE",
                        "actual_eqn",
                        "removed_terms",
                        "full_eqn",
                        ])


    #orders the coeffs into highest combination of powers first, stored as list of tuples
    order_by_pow_coeff = sorted(power_terms_dict.items(), key = lambda item: item[1], reverse = True)
    #turns list of tuples into a list, preserving order. Removing linear terms, as never want to delete those
    order_by_pow_coeff_no_linear = [i[0] for i in order_by_pow_coeff if i[1] > 1]
    order_by_pow_coeff_inc_linear = [i[0] for i in order_by_pow_coeff]

    MSE_array = []
    eqn_rm_array = []
    for terms in range(0,len(order_by_pow_coeff_no_linear)+1):
        removed = order_by_pow_coeff_no_linear[0:terms]
  
        eqn_rm = eqn
        for i in removed:
            eqn_rm = eqn_rm.subs(i, 0)
        eqn_rm_array.append(list(eqn_rm.as_coefficients_dict()))

        x, dx, A, w, p, c, A_val, p_val = symbols("x dx A w p c A_val p_val")
        current_eqn_lambda = lambdify([x, dx, A, w, p, c, A_val, p_val], eqn_rm)
        current_pred = current_eqn_lambda(voltage, dv_dt, A_dat, w_dat, p_dat, c_dat, A_val_dat, p_val_dat)

        current_MSE = calculate_MSE(current, current_pred)
        MSE_array.append(current_MSE)

        with open(os.path.join(output_filepath, f"eqn_{eqn_number}", f"summary_eqn_{eqn_number}.csv"), "a", newline='') as file:
            writer = csv.writer(file)
            writer.writerow([terms,
                            current_MSE,
                            eqn_rm,
                            removed,
                            eqn,
                            ]) 
        
        Path(os.path.join(output_filepath, f"eqn_{eqn_number}",f"removed_{terms}_terms")).mkdir(parents=True, exist_ok=True)
        generate_plots(time_data, current, dv_dt, current_pred, eqn_number, terms, output_filepath)
        generate_fft_graphs(time_data, current, current_pred, eqn_number, terms, output_filepath)               

    MSE_log_array = np.log10(np.array(MSE_array))
    highest_coeff_var = []
    for coeff_array_index in range(0, len(eqn_rm_array)):
        coeff_array = eqn_rm_array[coeff_array_index]
        coeff_array = [term for term in order_by_pow_coeff_inc_linear[::-1] if term in coeff_array]
        highest_coeff_var.append(coeff_array[-1])
        for coeff_index in range(0,len(coeff_array)):
            coeff_array[coeff_index] = "".join([str(coeff_array[coeff_index]), "\n"])
        eqn_rm_array[coeff_array_index] = coeff_array
        eqn_rm_array[coeff_array_index].insert(0,"".join([str(coeff_array_index),"\n\n"]))
        eqn_rm_array[coeff_array_index] = "".join(eqn_rm_array[coeff_array_index])

    highest_coeff_num = [sum(list(i.as_powers_dict().values())) for i in highest_coeff_var]
    #axs_overall.plot(highest_coeff_num, MSE_log_array, label = f"Eqn {eqn_number}")

    fig, axs = plt.subplots()
    axs.plot(range(0,len(MSE_log_array)), MSE_log_array, color = "red")
    axs.set_xlabel("Number of terms removed\n Terms remaining")
    axs.set_ylabel("log10 MSE")
    axs.set_title(f"Eqn-{eqn_number}. MSE by removing terms")
    axs.set_xticks(range(0,len(MSE_log_array)),eqn_rm_array)

    fig.tight_layout()
    plt.savefig(os.path.join(output_filepath, f"eqn_{eqn_number}", f"Eqn-{eqn_number}-MSE-by-terms.png"))
    plt.close(fig.figure)

""" axs_overall.legend()
fig_overall.tight_layout()
#fig_overall.savefig(os.path.join(output_filepath, f"{Hz}Hz-MSE-vs-power.png")) """







