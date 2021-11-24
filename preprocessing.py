import numpy as np
import pandas as pd
import pywt
import scipy.io
from scipy.ndimage.interpolation import zoom

from utils import *



mat_file = r'data/mill/mill.mat'
mat = scipy.io.loadmat(mat_file)
mat = mat['mill'][0]

column_names = ['case', 'run', 'VB', 'time', 'DOC', 'feed', 'material', 'smcAC', 'smcDC', 'vib_table', 'vib_spindle', 'AE_table', 'AE_spindle']

print('Loading data (.mat file)...')
dataset = []
for row in mat:
    df = pd.DataFrame(columns=column_names)
    sample_number = len(row[-1])
    for i, data in enumerate(row):
        if len(data) > 1:
            df[column_names[i]] = data.flatten()
        else:
            df[column_names[i]] = np.repeat(data.flatten(), sample_number)
    dataset.append(df)
print('Data loading completed.')
print('Info of the first dataframe:')
print(dataset[0].info())

totalscale = 256
wavename = 'morl'
sampling_rate = 250
resize_size = 224

fc = pywt.central_frequency(wavename)
cparam = 2 * fc * totalscale
scales = cparam / np.arange(totalscale, 1, -1)

print('Preprocessing data...')

processed_dataset = []
for i, df in enumerate(dataset):
    processed_data = np.empty((6, resize_size, resize_size))
    for j, column_name in enumerate(df.columns[-6:]):
        signal = df[column_name]
        signal = signal[(len(signal) - 1024) // 2:(len(signal) + 1024) // 2]
        cwtmatr, frequencies = pywt.cwt(signal, scales, wavename, 1 / sampling_rate)

        resized = scipy.ndimage.zoom(cwtmatr, (resize_size / cwtmatr.shape[0], resize_size / cwtmatr.shape[1]), order = 3)
        processed_data[j] = resized

    print(f'\rContinuous wavelet transform... ({i}/{len(dataset)})', end='')
    processed_dataset.append(processed_data)
print(f'\rContinuous wavelet transform completed.')
