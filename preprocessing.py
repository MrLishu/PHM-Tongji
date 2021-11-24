import numpy as np
import pandas as pd
import pywt
import scipy.io



print('Loading data (.mat file)...')
mat_file = r'data/mill/mill.mat'
mat = scipy.io.loadmat(mat_file)
mat = mat['mill'][0]

condition_name = ['case', 'run', 'VB', 'time', 'DOC', 'feed', 'material'] 
signal_name = ['smcAC', 'smcDC', 'vib_table', 'vib_spindle', 'AE_table', 'AE_spindle']

condition_data = np.array(mat[condition_name].tolist()).squeeze()
condition_dataset = pd.DataFrame(condition_data, columns=condition_name)
signal_dataset = [pd.DataFrame(np.array(data).squeeze().T, columns=signal_name) for data in mat[signal_name].tolist()]
print('Data loading completed.')
print('Info of the condition dataset and the first signal dataframe:')
print(condition_dataset.info())
print(signal_dataset[0].info())

totalscale = 256
wavename = 'morl'
sampling_rate = 250
resample_number = 1024
resize_size = 224

fc = pywt.central_frequency(wavename)
cparam = 2 * fc * totalscale
scales = cparam / np.arange(totalscale, 0, -1)

print('Preprocessing data...')
cwt_dataset = np.empty((len(signal_dataset), len(signal_name), totalscale, resample_number))
for i, df in enumerate(signal_dataset):
    data = df.to_numpy().T
    sample_number = data.shape[1]
    data = data[:, (sample_number - resample_number) // 2:(sample_number + resample_number) // 2]
    cwtmatr, frequencies = pywt.cwt(data, scales, wavename, 1 / sampling_rate)
    cwt_dataset[i] = cwtmatr.transpose(1, 0, 2)
    print(f'\rContinuous wavelet transform... ({i}/{len(signal_dataset)})', end='')
print(f'\nContinuous wavelet transform completed.')
