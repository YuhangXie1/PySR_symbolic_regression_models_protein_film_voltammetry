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

    file_loc = rf"results/20250513-fit-workflow/"
    voltage_eqn_file = pd.read_csv(os.path.join(file_loc, str(Hz), "summary_voltage_model.csv"))
    voltage_sympy = sympify(np.array(voltage_eqn_file["picked_equation"])[0])

    x0 = Symbol("x0")
    dv_dt_func = diff(voltage_sympy, x0)
    dv_dt_lambda = lambdify(x0,dv_dt_func)
    dv_dt = dv_dt_lambda(time_data)

    #loading summary file
    #files_freq = [9, 36, 45, 54, 63, 72, 81, 90, 99]
    data = pd.read_csv(os.path.join(file_loc, str(Hz),"summary_current_model.csv"))
    data_equations = sympify(np.array(data["substituted_form"]))

    #equation_coeffs_dict = [eqns.as_coefficients_dict() for eqns in data_equations]

    return time_data, voltage, current, dv_dt, data_equations


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
    plt.savefig(os.path.join(output_filepath, f"eqn_{eqn_number}", str(terms), f"Eqn-{eqn_number}-rm-{terms}-current-time.png"))
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
    plt.savefig(os.path.join(output_filepath, f"eqn_{eqn_number}", str(terms), f"Eqn-{eqn_number}-rm-{terms}-dv_dt.png"))
    plt.close(fig.figure)
    

##main##
#files_freq = [9, 36, 45, 54, 63, 72, 81, 90, 99]
Hz = 99
time_data, voltage, current, dv_dt, data_equations = load_data(Hz)

fig_overall, axs_overall = plt.subplots()
axs_overall.set_xlabel("Highest total power of variables")
axs_overall.set_ylabel("Log MSE")
axs_overall.set_title("All equations, highest power vs MSE")


overall_powers = []
overall_log_MSE = []

for eqn_number in range(0, len(data_equations)):
    eqn = data_equations[eqn_number]


    output_filepath = rf"results/20250515-equation-post/{Hz}/"
    Path(output_filepath, f"eqn_{eqn_number}").mkdir(parents=True, exist_ok=True)
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
    print(eqn_coeff_list)

    order_by_pow_coeff= {}
    for i in eqn_coeff_list:
        powers_dict = i.as_powers_dict()
        total_power = sum(list(powers_dict.values()))
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
        print(removed)
        eqn_rm = eqn
        for i in removed:
            eqn_rm = eqn_rm.subs(i, 0)
        eqn_rm_array.append(list(eqn_rm.as_coefficients_dict()))

        eqn_lambda = lambdify([x,dx],eqn_rm)
        current_pred = eqn_lambda(voltage, dv_dt)
        current_MSE = calculate_MSE(current, current_pred)
        MSE_array.append(current_MSE)


        Path(os.path.join(output_filepath, f"eqn_{eqn_number}",str(terms))).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(output_filepath, f"eqn_{eqn_number}", f"summary_eqn_{eqn_number}.csv"), "a", newline='') as file:
            writer = csv.writer(file)
            writer.writerow([terms,
                            current_MSE,
                            eqn_rm,
                            removed,
                            eqn,
                            ]) 
                        
        generate_plots(time_data, current, dv_dt, current_pred, eqn_number, terms, output_filepath)               

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
plt.savefig(os.path.join(output_filepath, f"{Hz}Hz-MSE-vs-power.png"))

""" pairwise_comparison = set([tuple(sorted(i)) for i in list(product(range(0,len(equation_coeffs_dict)),range(0,len(equation_coeffs_dict)))) if i[0] != i[1]])


all_keys = []
for d in equation_coeffs_dict:
    for i in d.keys():
        all_keys.append(i)
all_keys_no_duplicate = set(all_keys)

term_similarity_df = pd.DataFrame(index= range(0,len(equation_coeffs_dict)), columns= range(0, len(equation_coeffs_dict)))

for pair in pairwise_comparison:
    dict_one = set(equation_coeffs_dict[pair[0]])
    dict_two = set(equation_coeffs_dict[pair[1]])

    shared_terms = dict_one.intersection(dict_two)

    term_similarity_df.at[pair[0],pair[1]] = int(len(shared_terms)/len(all_keys_no_duplicate)*100)

print(term_similarity_df)

for key in all_keys_no_duplicate:
    for d in equation_coeffs_dict:
        if key not in d:
            d[key] = 0


coeff_similarity_df = pd.DataFrame(index= range(0,len(equation_coeffs_dict)), columns= range(0, len(equation_coeffs_dict)))

for pair in pairwise_comparison:
    dict_one = equation_coeffs_dict[pair[0]]
    dict_two = equation_coeffs_dict[pair[1]]

    x = np.array([float(i) for i in dict_one.values()])
    y = np.array([float(i) for i in dict_two.values()])

    coeff_similarity_df.at[pair[0],pair[1]] = cos_similarity(x,y)
    #pair[0] is row number. pair[1] is column number

print(coeff_similarity_df) """
