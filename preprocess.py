import numpy as np
import pywt



def resample(raw_dataset, sampling_rate, resample_number=1024, step=1, save=False, save_filepath=None):
    resample_dataset = np.empty((len(raw_dataset), len(raw_dataset[0].columns), resample_number))
    for i, df in enumerate(raw_dataset):
        data = df.to_numpy().T
        data -= data.mean(axis=-1)[:, np.newaxis]
        data /= data.std(axis=-1)[:, np.newaxis]
        sample_number = data.shape[1]
        data = data[:, (sample_number - resample_number * step) // 2:(sample_number + resample_number * step) // 2:step]

        resample_dataset[i] = data
        print(f'\rResampling signals... ({i + 1}/{len(raw_dataset)})', end='')
    print(f'\nResampling completed.')

    if save:
        np.save(save_filepath, resample_dataset)
        print(f'Data saved at {save_filepath}')
    
    return resample_dataset, sampling_rate / step

def cwt(resample_dataset, sampling_rate, totalscale=256, wavename='morl', save=False, save_filepath=None):
    fc = pywt.central_frequency(wavename)
    cparam = 2 * fc * totalscale
    scales = cparam / np.arange(totalscale, 0, -1)

    cwt_dataset = np.empty((resample_dataset.shape[0], resample_dataset.shape[1], totalscale, resample_dataset.shape[2]))

    for i, data in enumerate(resample_dataset):
        cwtmatr, frequencies = pywt.cwt(data, scales, wavename, 1 / sampling_rate)
        cwt_dataset[i] = cwtmatr.transpose(1, 0, 2)
        print(f'\rContinuous wavelet transform... ({i + 1}/{resample_dataset.shape[0]})', end='')
    print(f'\nContinuous wavelet transform completed.')

    if save:
        np.save(save_filepath, cwt_dataset)
        print(f'Data saved at {save_filepath}')

    return cwt_dataset
