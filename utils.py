import numpy as np
import matplotlib.pyplot as plt



def plotCwt(sampling_rate, signal, cwtmatr, frequencies):
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

def plotSignal(signal, sampling_rate=1):
    t = np.arange(len(signal)) / sampling_rate
    plt.figure(figsize=(8, 4))
    plt.plot(t, signal)
    plt.xlabel(u"time(s)")
    plt.title(u"Signal")