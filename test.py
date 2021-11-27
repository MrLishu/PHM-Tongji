import os
import numpy as np
import pandas as pd
import pywt

from utils import *



data_directory = r'data/c1'
dataframe_number = 3  # max=315
signal_name = ['Fx', 'Fy', 'Fz', 'Ax', 'Ay', 'Az', 'AE_rms']
sampling_rate = 5000

raw_dataset = []
for i in range(dataframe_number):
    df = pd.read_csv(os.path.join(data_directory, f'c_1_{i + 1:03d}.csv'), names=signal_name)
    raw_dataset.append(df)
    print(f'\rLoading data... ({i + 1}/{dataframe_number})', end='')
print(f'\nData loading completed.')

resample_number = 1024
step = 50

totalscale = 256
wavename = 'morl'

fc = pywt.central_frequency(wavename)
cparam = 2 * fc * totalscale
scales = cparam / np.arange(totalscale, 0, -1)

print('Preprocessing data...')
resample_dataset = np.empty((dataframe_number, len(signal_name), resample_number))
cwt_dataset = np.empty((dataframe_number, len(signal_name), totalscale, resample_number))

for i, df in enumerate(raw_dataset):
    data = df.to_numpy().T
    sample_number = data.shape[1]
    data = data[:, (sample_number - resample_number * step) // 2:(sample_number + resample_number * step) // 2:step]
    cwtmatr, frequencies = pywt.cwt(data, scales, wavename, 1 / (sampling_rate / step))

    resample_dataset[i] = data
    cwt_dataset[i] = cwtmatr.transpose(1, 0, 2)
    print(f'\rContinuous wavelet transform... ({i + 1}/{dataframe_number})', end='')
print(f'\nContinuous wavelet transform completed.')

label_path = r'data/c1_wear.csv'
label_dataset = pd.read_csv((label_path))
print(label_dataset.info())

label_dataset = label_dataset.to_numpy()