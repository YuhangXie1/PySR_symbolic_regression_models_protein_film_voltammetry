import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
loc=r"C:\Users\Longs\Coding\rotation_project\Data_for_eq_learning"
import os
files=os.listdir(loc)
reduce=1
band_size=0.1
desired_harmonic=3
for file in files[:1]:
    if "FTacV" in file:
        continue
    if "Hz" in file:
        split_file=file.split("_") 
        freq=split_file[:2]
        current_file="_".join(freq+["2", "cv","current"])
        potential_file="_".join(freq+["2", "cv","voltage"])
        current_file_data=np.loadtxt(os.path.join(loc, current_file))
        current=current_file_data[::reduce,1]
        time=current_file_data[::reduce, 0]
        potential=np.loadtxt(os.path.join(loc, potential_file))[::reduce,1]
        
        plt.plot(potential, current, label=" ".join(freq))

        ft=np.fft.fft(current)
        freqs=np.fft.fftfreq(len(time), time[1]-time[0])
        freq_loc=desired_harmonic*int(freq[0])
        new_ft=np.zeros(len(ft), dtype="complex")
        for sign in [-1, 1]:
            ft_loc=np.where((freqs>sign*freq_loc-(band_size*int(freq[0]))) & (freqs<sign*freq_loc+(band_size*int(freq[0]))))
            new_ft[ft_loc]=ft[ft_loc]

        #plt.plot(freqs, ft, label=freq)
        #plt.plot(freqs, new_ft)
        inversefft=np.fft.ifft(new_ft)
        #plt.plot(time, inversefft)


plt.legend()
plt.show()