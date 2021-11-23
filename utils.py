import os
import scipy.signal as signal
import numpy as np
import pandas as pd
from sklearn import preprocessing
import pywt
import scipy
import scipy.io
from scipy.ndimage.interpolation import zoom
from scipy.fftpack import fft
import matplotlib.pyplot as plt


def plot_cwt(sampling_rate, signal, cwtmatr, frequencies):
    t = np.arange(len(signal)) / sampling_rate
    plt.figure(figsize=(8, 4))
    plt.subplot(211)
    plt.plot(t, signal)
    plt.xlabel(u"time(s)")
    plt.title(u"Time spectrum")
    plt.subplot(212)
    plt.contourf(t, frequencies, abs(cwtmatr))
    plt.ylabel(u"freq(Hz)")
    plt.xlabel(u"time(s)")
    plt.subplots_adjust(hspace=0.4)
    plt.show()