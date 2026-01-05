import os
import numpy as np
import pandas as pd
from sympy import Symbol, symbols, sympify, diff
from sympy.utilities import lambdify
import matplotlib.pyplot as plt
from pathlib import Path
import csv
from scipy.signal import find_peaks
from scipy.optimize import least_squares

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

def load_data(Hz):
    """
    Returns numpy arrays of time_data, voltage, current and frequency for a set of data files
    """
    #loading data
    data_volt = pd.read_csv(f"Data_for_eq_learning/FTacV_before_PSV_cv_current", sep="\t", names = ["time","voltage"])
    data_amp = pd.read_csv(f"Data_for_eq_learning/FTacV_before_PSV_cv_voltage", sep="\t", names = ["time","current"])
    data_combined = data_volt.join(data_amp["current"])

    #slicing data to remove start and end noise. Putting data into tuples
    slice_start = 550
    slice_end = -100
    time_data = np.array(data_combined["time"].iloc[slice_start:slice_end])
    voltage = np.array(data_combined["voltage"].iloc[slice_start:slice_end])
    current = np.array(data_combined["current"].iloc[slice_start:slice_end])

    return time_data, voltage, current, [slice_start, slice_end]

def filter_fft_by_freqs(freqs, ft, center_freq, band_width_hz):
    """
    Keeps FFT bins within center_freq +/- band_width_hz, zeros the rest.
    Works with negative frequencies automatically because freqs contains both signs.
    """
    mask = (freqs >= (center_freq - band_width_hz)) & (freqs <= (center_freq + band_width_hz))
    filtered = np.zeros_like(ft, dtype=complex)
    filtered[mask] = ft[mask]
    return filtered

def remove_first_harmonic(time_data, current, freqs, ft, fundamental_freq, band_width):
    """
    Returns (fundamental_time, residual_time, ft_first, ft_residual)
    - fundamental_time: reconstructed first-harmonic in time domain (real-valued)
    - residual_time: current minus fundamental_time
    - ft_first: FFT containing only the fundamental bins
    - ft_residual: FFT of the residual
    """
    # 1) create filtered FFT that keeps only the fundamental (±band_width)
    ft_first = filter_fft_by_freqs(freqs, ft, fundamental_freq, band_width)

    # 2) inverse to time domain and make sure result is real
    fundamental_time = np.real(np.fft.ifft(ft_first))

    # 3) subtract from original to get residual (higher harmonics)
    residual_time = current - fundamental_time

    # 4) FFT of residual if you need it
    ft_residual = np.fft.fft(residual_time)

    return fundamental_time, residual_time, ft_first, ft_residual

def estimate_ac_frequency(time_data, signal,
                          do_detrend=True, window=True, zero_pad_factor=4,
                          search_max_harmonic=None, peak_search_width_hz=None):
    """
    Estimate the AC drive frequency from time_data and signal (current or voltage).
    Returns: dict with keys: 'f_est', 'f_refined', 'method', 'fs', 'notes'
    
    Steps:
      1) Detrend (optional) -> window (optional) -> zero-pad
      2) FFT and find peaks in positive-frequency spectrum
      3) Choose the strongest low-frequency peak (not DC), refine with parabola
      4) Cross-check with autocorrelation period estimate
      5) Optionally refine using nonlinear least-squares sine fit
    """
    # basic checks
    N = len(signal)
    if N < 10:
        raise ValueError("signal too short")
    dt = time_data[1] - time_data[0]
    fs = 1.0 / dt

    # 0) preprocess
    y = signal.astype(float).copy()
    if do_detrend:
        xidx = np.arange(N)
        p = np.polyfit(xidx, y, 1)
        y = y - np.polyval(p, xidx)
    if window:
        win = np.hanning(N)
        y = y * win

    # zero pad
    Npad = int(N * zero_pad_factor)
    if Npad <= N:
        Npad = N
    ypad = np.pad(y, (0, Npad - N))

    # FFT
    freqs = np.fft.fftfreq(len(ypad), d=dt)
    ft = np.fft.fft(ypad)
    pos_mask = freqs > 0
    freqs_p = freqs[pos_mask]
    mag_p = np.abs(ft[pos_mask])

    # avoid DC: set small freqs to zero temporarily
    # choose lower bound > 0.5 * (1/T) to ignore DC / very low drift
    df = freqs_p[1] - freqs_p[0]
    low_freq_threshold = max(df*2, 0.1)  # 0.1 Hz floor; adjust if needed
    valid_mask = freqs_p > low_freq_threshold
    if not np.any(valid_mask):
        valid_mask = freqs_p > 0

    # find peaks in positive spectrum above threshold
    peaks, props = find_peaks(mag_p[valid_mask], height=np.max(mag_p[valid_mask])*0.05)  # peaks above 5% of max
    if len(peaks) == 0:
        # fallback: take largest bin (excluding DC)
        candidate_idx = np.argmax(mag_p)
        f_bin = freqs_p[candidate_idx]
        method = "largest_bin_fallback"
    else:
        # map peak indices back to freqs_p indices
        valid_indices = np.where(valid_mask)[0]
        peak_indices = valid_indices[peaks]

        # pick the peak with highest amplitude among low-to-moderate freqs
        # optionally restrict to search_max_harmonic * expected f if you had an estimate
        # here choose the *lowest* high peak: often fundamental is the lowest strong peak
        sorted_peaks = sorted(peak_indices, key=lambda i: freqs_p[i])  # ascending freq
        # pick first peak that is reasonably strong
        candidate_idx = None
        for idx in sorted_peaks:
            # require that peak is reasonably above noise (peak height check)
            if mag_p[idx] > (np.mean(mag_p) + 2*np.std(mag_p)):
                candidate_idx = idx
                break
        if candidate_idx is None:
            candidate_idx = peak_indices[np.argmax(mag_p[peak_indices])]
        method = "fft_peak"

    # quadratic interpolation (parabolic fit in log magnitude) for sub-bin refinement
    def quadratic_refine(freqs_arr, mag_arr, k):
        if k <= 0 or k >= len(mag_arr)-1:
            return freqs_arr[k], mag_arr[k]
        y0 = np.log(mag_arr[k-1] + 1e-30)
        y1 = np.log(mag_arr[k]   + 1e-30)
        y2 = np.log(mag_arr[k+1] + 1e-30)
        denom = (y0 - 2*y1 + y2)
        if denom == 0:
            return freqs_arr[k], mag_arr[k]
        delta = 0.5 * (y0 - y2) / denom
        df_local = freqs_arr[1] - freqs_arr[0]
        f_ref = freqs_arr[k] + delta * df_local
        mag_ref = np.exp(y1 - 0.25*(y0 - y2)*delta)
        return f_ref, mag_ref

    f_bin = freqs_p[candidate_idx]
    f_refined, mag_refined = quadratic_refine(freqs_p, mag_p, candidate_idx)

    # 2) autocorrelation check (gives period)
    # compute on original (non-windowed) detrended signal for period estimation
    y_check = signal.copy().astype(float)
    if do_detrend:
        xidx = np.arange(len(y_check))
        p = np.polyfit(xidx, y_check, 1)
        y_check = y_check - np.polyval(p, xidx)
    # autocorrelation via FFT
    ac = np.fft.ifft(np.abs(np.fft.fft(y_check, n=2*len(y_check)))**2).real
    ac = ac[:len(y_check)]
    ac[0] = 0  # ignore zero lag
    ac_peak_idx = np.argmax(ac)
    # find first significant peak (not at lag zero)
    # search for first local maximum after lag > 1
    # get a few top lags and compute corresponding frequencies
    candidate_lags = np.argsort(ac)[-5:]  # top 5 lags
    candidate_lags = candidate_lags[candidate_lags>1]
    if len(candidate_lags) > 0:
        # choose the smallest lag among candidates (fastest period)
        best_lag = np.min(candidate_lags)
        f_ac = 1.0 / (best_lag * dt)
    else:
        # fallback: use sample autocorr peak index
        if ac_peak_idx > 1:
            f_ac = 1.0 / (ac_peak_idx * dt)
        else:
            f_ac = None

    # 3) optional least squares sine refinement (gives freq, amp, phase)
    # initial guess: freq = f_refined, amplitude from mag_refined
    def refine_with_sine(time, y, f0):
        # fit A*sin(2π f t + phi) + offset using least squares
        if f0 is None or f0 <= 0 or f0 >= fs/2:
            return None
        # model params: [A, phi, c]  (offset c)
        def residuals(params):
            A, phi, c = params
            model = A * np.sin(2*np.pi*f0*time + phi) + c
            return (model - y)
        # initial guess
        A0 = (np.max(y) - np.min(y))/2
        phi0 = 0.0
        c0 = 0.0
        try:
            res = least_squares(residuals, x0=[A0, phi0, c0], xtol=1e-8, ftol=1e-8)
            A_est, phi_est, c_est = res.x
            # compute residual RMS; if high the fit is poor
            rss = np.sqrt(np.mean(res.fun**2))
            return {'f': f0, 'A': A_est, 'phi': phi_est, 'c': c_est, 'rms': rss}
        except Exception:
            return None

    sine_ref = refine_with_sine(time_data, signal, f_refined)

    # decide final estimate: prefer fft_refined unless autocorr strongly disagrees
    final_f = f_refined
    method_final = method
    notes = []
    if f_ac is not None:
        # compare
        reldiff = abs(f_ac - f_refined) / f_refined
        if reldiff < 0.02:  # within 2% agreement
            notes.append(f"autocorr agrees: {f_ac:.6f} Hz (rel diff {reldiff:.3%})")
        else:
            notes.append(f"autocorr {f_ac:.6f} Hz disagrees with fft {f_refined:.6f} Hz (rel diff {reldiff:.3%})")
            # still keep fft, but mark disagreement
    else:
        notes.append("autocorr unavailable")

    if sine_ref is not None:
        notes.append(f"sine fit RMS={sine_ref['rms']:.3e}, amp={sine_ref['A']:.3e}")
        # optionally accept slight refinement if it reduces RMS (we don't vary f here)
    else:
        notes.append("sine fit unavailable or failed")

    # safety: ensure final_f < Nyquist
    if final_f >= fs/2:
        notes.append(f"warning: estimated frequency {final_f:.3f} >= Nyquist {fs/2:.3f} -> aliasing risk")

    return {
        'f_est_bin': f_bin,
        'f_refined': final_f,
        'magnitude': float(mag_refined),
        'fs': fs,
        'method': method_final,
        'autocorr_freq': f_ac,
        'sine_fit': sine_ref,
        'notes': notes
    }



##main##
#files_freq = [9, 36, 45, 54, 63, 72, 81, 90, 99]
files_freq = [9, 36, 45, 54, 63, 72, 81, 90, 99]
#files_freq = [0]
Hz = 9

time_data, voltage, current, slice = load_data(Hz)

harmonics_lines = [k*Hz for k in range(1,10)]

band_size=0.5
freqs, ft = fourier_transform(current, time_data)

#plotting all harmonics
fig, axs = plt.subplots()
axs.plot(freqs, np.log10(ft**2), color = "blue", label="real")
#axs.scatter(harmonics_lines, np.zeros(len(harmonics_lines)), color = "red", label="harmonics", alpha = 0.5)
axs.set_xlabel("Frequency")
axs.set_ylabel("Log10 ft^2")
axs.set_title(f"File FTV, all harmonics")
axs.legend()
fig.tight_layout()
plt.show()

""" #plot a select harmonic
fig, axs = plt.subplots()
harmonic = 11

ft_filtered = filter_fft(freqs, ft, band_size, Hz, harmonic)
log_ft_filtered = np.log10(ft_filtered**2)

axs.plot(freqs, log_ft_filtered, color = "blue", label="real")
axs.set_xlim(harmonic*int(Hz)-harmonic*int(Hz)*0.03, harmonic*int(Hz)+harmonic*int(Hz)*0.03)
axs.legend()
axs.set_xlabel("Frequency (Hz)")
axs.set_ylabel("Log10 ft^2")
axs.set_title(f"File FTV {Hz}Hz harmonic {harmonic} in freq domain")
fig.tight_layout()
plt.show() """

filtered_fft_1 = filter_fft(freqs, ft, band_size, Hz, 1)

fig, axs = plt.subplots()
axs.plot(freqs, np.log10(filtered_fft_1**2), color = "blue", label="real")
axs.set_xlabel("Frequency")
axs.set_ylabel("Log10 ft^2")
axs.set_title(f"File FTV, Harmonic 1")
axs.legend()
fig.tight_layout()
plt.show()

inverse_filtered_fft_1 = np.fft.ifft(filtered_fft_1)

fig, axs = plt.subplots()
axs.plot(time_data, inverse_filtered_fft_1, color = "blue", label="real")
axs.set_xlabel("Time (s)")
axs.set_ylabel("Current (A)")
axs.set_title(f"File FTV, Harmonic 1, inverse. Time domain")
axs.legend()
fig.tight_layout()
plt.show()

current_without_harmonic_1 = current - inverse_filtered_fft_1

fig, axs = plt.subplots()
axs.plot(time_data, current_without_harmonic_1, color = "blue", label="real")
axs.set_xlabel("Time (s)")
axs.set_ylabel("Current (A)")
axs.set_title(f"File FTV, All harmonics except 1. Time domain")
axs.legend()
fig.tight_layout()
plt.show()

fft_without_harmonic_1_freqs,  fft_without_harmonic_1_ft = fourier_transform(current_without_harmonic_1, time_data)

fig, axs = plt.subplots()

axs.plot(fft_without_harmonic_1_freqs, np.log10(fft_without_harmonic_1_ft**2), color = "blue", label="real")
axs.set_xlabel("Frequency")
axs.set_ylabel("Log10 ft^2")
axs.set_title(f"File FTV, FFT without harmonic 1")
axs.legend()
fig.tight_layout()
plt.show()


""" import numpy as np
from scipy.signal import find_peaks

# prepare positive freqs and power
dt = time_data[1] - time_data[0]
fs = 1.0 / dt
N = len(time_data)
ft_full = ft
freqs_full = freqs

pos = freqs_full > 0
freqs_p = freqs_full[pos]
mag_p = np.abs(ft_full[pos])              # magnitude
power_p = mag_p**2                        # power
log_power_p = np.log10(power_p + 1e-30)   # avoid log(0)

# function to refine peak by quadratic interpolation in log-power
def refine_peak(freqs_arr, power_arr, idx):
    if idx <= 0 or idx >= len(power_arr)-1:
        return freqs_arr[idx], power_arr[idx]
    y0 = np.log(power_arr[idx-1] + 1e-30)
    y1 = np.log(power_arr[idx]   + 1e-30)
    y2 = np.log(power_arr[idx+1] + 1e-30)
    denom = (y0 - 2*y1 + y2)
    if denom == 0:
        return freqs_arr[idx], power_arr[idx]
    delta = 0.5 * (y0 - y2) / denom
    df_local = freqs_arr[1] - freqs_arr[0]
    f_ref = freqs_arr[idx] + delta * df_local
    p_ref = np.exp(y1 - 0.25*(y0 - y2)*delta)
    return f_ref, p_ref

# detect harmonics and compute SNR
harmonics = list(range(1, 13))  # pick up to 12th harmonic for check
results = []
noise_floor = np.median(power_p)  # simple noise estimate; you may want robust measure

search_halfwidth_hz = 0.5  # search +/- 0.5 Hz (adjust if band is wider)
for k in harmonics:
    f_expected = k * Hz
    if f_expected >= fs/2:
        results.append((k, f_expected, None, None, 'above_nyquist'))
        continue
    # mask for search window
    mask = (freqs_p >= f_expected - search_halfwidth_hz) & (freqs_p <= f_expected + search_halfwidth_hz)
    if not np.any(mask):
        results.append((k, f_expected, None, None, 'no_bins'))
        continue
    # pick local maximum in window
    local_idx = np.argmax(power_p[mask])
    # map to global pos-array index
    indices = np.where(pos)[0]  # mapping pos->original
    # index in freqs_p
    idx_in_p = np.where(mask)[0][0] + local_idx
    f_ref, p_ref = refine_peak(freqs_p, power_p, idx_in_p)
    snr = p_ref / (noise_floor + 1e-30)
    results.append((k, f_expected, f_ref, p_ref, snr))

# print summary
print("k, f_expected, f_refined, power, SNR")
for r in results:
    print(r)

# Plot with markers at the actual power levels
fig, ax = plt.subplots()
ax.plot(freqs_p, log_power_p, color='blue', label='real (positive freqs)')
# markers at harmonic peaks (log scale)
for k, f_exp, f_ref, p_ref, snr in results:
    if f_ref is None:
        continue
    ax.scatter([f_ref], [np.log10(p_ref + 1e-30)], color='red', zorder=5)
    ax.text(f_ref, np.log10(p_ref + 1e-30), f'{k}', color='red', fontsize=8, va='bottom')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Log10 power')
ax.set_title('Harmonic markers at actual peak positions')
ax.set_xlim(0, min( max(freqs_p), 12*Hz + 20 ))
plt.show() """
