import os
import numpy as np
import pandas as pd
from sympy import Symbol, symbols, sympify, diff
from sympy.utilities import lambdify
from itertools import product
import matplotlib.pyplot as plt
from pathlib import Path
import csv


def cos_similarity(x,y):
    return np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))

def calculate_MSE(y, y_pred):
    """Returns the mean squared error between a y value (real) and a predicted y value (prediction)."""
    mse = sum((y_pred - y)**2)
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

def load_data(Hz):
    #loading data
    data_volt = pd.read_csv(f"Data_for_eq_learning/{Hz}_Hz_2_cv_voltage", sep="\t", names = ["time","voltage"])
    data_amp = pd.read_csv(f"Data_for_eq_learning/{Hz}_Hz_2_cv_current", sep="\t", names = ["time","current"])
    data_combined = data_volt.join(data_amp["current"])

    #slicing data to remove start and end noise. Putting data into tuples
    slice_start = 550
    slice_end = -100
    time_data = np.array(data_combined["time"].iloc[slice_start:slice_end])
    voltage = np.array(data_combined["voltage"].iloc[slice_start:slice_end])
    current = np.array(data_combined["current"].iloc[slice_start:slice_end])

    #loading voltage equation
    file_loc = rf"results/20250513-fit-workflow/"
    voltage_eqn_file = pd.read_csv(os.path.join(file_loc, str(Hz), "summary_voltage_model.csv"))
    voltage_sympy = sympify(np.array(voltage_eqn_file["picked_equation"])[0])

    x0 = Symbol("x0")
    dv_dt_func = diff(voltage_sympy, x0)
    dv_dt_lambda = lambdify(x0,dv_dt_func)
    dv_dt = dv_dt_lambda(time_data)

    #loading summary file
    data = pd.read_csv(os.path.join(file_loc, str(Hz),"summary_current_model.csv"))
    data_equations = sympify(np.array(data["substituted_form"]))

    return time_data, voltage, current, dv_dt, data_equations

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
    axs.set_xlim(0,400)
    axs.set_ylim(-4, 3)
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


##main##
#files_freq = [9, 36, 45, 54, 63, 72, 81, 90, 99]
Hz = 3
load_filepath = rf"results\20250620-multi-fit-workflow-9\time-shift-normalised_dv_dt_aval"
output_filepath = rf"results\20250616-CjX-fit-workflow-2\CjX\20250611-blank-PSV-3Hz\fft"
file_loc = rf"results\20250616-CjX-fit-workflow-2\blank\20250611-blank-PSV-3Hz"

time_data, voltage, current, slice = load_data_2(load_filepath)

#loading voltage equation
voltage_eqn_file = pd.read_csv(os.path.join(file_loc, "summary_voltage_model.csv"))
voltage_sympy = sympify(np.array(voltage_eqn_file["voltage_eqn"])[0])

x0 = Symbol("x0")
dv_dt_eqn = sympify(np.array(voltage_eqn_file["dv_dt_eqn"])[0])
dv_dt_lambda = lambdify(x0,dv_dt_eqn)
dv_dt = dv_dt_lambda(time_data)

#loading summary file
data = pd.read_csv(os.path.join(file_loc,"summary_current_model.csv"))
data_equations = sympify(np.array(data["substituted_form"]))

####

fig_overall, axs_overall = plt.subplots()
axs_overall.set_xlabel("Highest total power of variables")
axs_overall.set_ylabel("Log MSE")
axs_overall.set_title("All equations, highest power vs MSE")

overall_powers = []
overall_log_MSE = []

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

    x, dx = symbols("x dx")
    eqn_coeff_list = list(eqn.as_coefficients_dict())

    order_by_pow_coeff= {}
    for i in eqn_coeff_list:
        powers_dict = i.as_powers_dict()
        total_power = 0
        for key, value in powers_dict.items():
            if key == x or key == dx:
                total_power += value
        order_by_pow_coeff[i] = total_power

    #orders the coeffs into highest combination of powers first, stored as list of tuples
    order_by_pow_coeff = sorted(order_by_pow_coeff.items(), key = lambda item: item[1], reverse = True)
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

        eqn_lambda = lambdify([x,dx],eqn_rm)
        current_pred = eqn_lambda(voltage, dv_dt)
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
    axs_overall.plot(highest_coeff_num, MSE_log_array, label = f"Eqn {eqn_number}")

    fig, axs = plt.subplots()
    axs.plot(range(0,len(MSE_log_array)), MSE_log_array, color = "red")
    axs.set_xlabel("Number of terms removed\n Terms remaining")
    axs.set_ylabel("log10 MSE")
    axs.set_title(f"Eqn-{eqn_number}. MSE by removing terms")
    axs.set_xticks(range(0,len(MSE_log_array)),eqn_rm_array)

    fig.tight_layout()
    plt.savefig(os.path.join(output_filepath, f"eqn_{eqn_number}", f"Eqn-{eqn_number}-MSE-by-terms.png"))
    plt.close(fig.figure)

axs_overall.legend()
fig_overall.tight_layout()
fig_overall.savefig(os.path.join(output_filepath, f"{Hz}Hz-MSE-vs-power.png"))
