# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 11:36:01 2025

@author: Acer
"""

import numpy as np
import pandas as pd
from numba import njit

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def rolling_mean_np(a, window):
    rw = rolling_window(a, window)
    return rw.mean(axis=-1)

def rolling_var_np(a, window):
    rw = rolling_window(a, window)
    return rw.var(axis=-1)

def rolling_mean_pd(df, window):
    return df.rolling(window=window).mean()

def rolling_var_pd(df, window):
    return df.rolling(window=window).var()

@njit
def rolling_mean_numba(a, window):
    n = a.shape[0]
    result = np.empty(n - window + 1)
    cumsum = 0.0 
    for i in range(window):
        cumsum += a[i]
    result[0] = cumsum / window
    for i in range (1, n -  window + 1):
        cumsum += a[i + window - 1] - a[i - 1]
        result[i] = cumsum / window
    return result

def ewma_np(x, alpha):
    result = np.empty_like(x)
    result[0] = x[0]
    for i in range(1, len(x)):
        result[i] = alpha * x[i] + (1 - alpha) * result[i - 1]
    return result

def ewma_pd(series, alpha):
    return series.ewm(alpha=alpha).mean()

def ew_cov_np(x, y, alpha):
    cov = np.empty_like(x)
    mean_x = x[0]
    mean_y = y[0]
    cov[0] = 0.0
    for i in range(1, len(x)):
        dx = x[i] - mean_x
        dy = y[i] - mean_y
        cov[i] = (1 - alpha) * (cov[i-1] + alpha * dx * dy)
        mean_x = (1 - alpha) * mean_x + alpha * x[i]
        mean_y = (1 - alpha) * mean_y + alpha * y[i]
    return cov

def fft_spectrum(x, fs):
    n = len(x)
    freq = np.fft.fftfreq(n, d=1/fs)
    spectrum = np.abs(np.fft.fft(x)) / n
    return freq[:n//2], spectrum[:n//2]

def bandpass_filter_fft(x, fs, lowcut, highcut):
    n = len(x)
    fft_vals = np.fft.fft(x)
    freqs = np.fft.fftfreq(n, d=1/fs)
    
    mask = (np.abs(freqs) >= lowcut) & (np.abs(freqs) <= highcut)
    fft_filtered = np.where(mask, fft_vals, 0)
    filtered = np.fft.ifft(fft_filtered).real
    return filtered

def auto_select_method(data_len):
    if data_len < 1e5:
        return 'pandas'
    if data_len < 5e6:
        return 'numpy'
    else:
        return 'numba'