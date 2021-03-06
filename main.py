import numpy as np
from loadData import loadPHMData
from preprocess import resample, cwt


raw_data_directory = r'data/raw/PHM2010'
processed_data_directory = r'data/processed/PHM2010'

sampling_rate=50000

resample_number=1024
step=1

totalscale=256
wavename='morl'

# (c1_raw, c1_label), = loadPHMData(data_directory=raw_data_directory, dataset_names=['c1'])
# c1_resample, rs_rate = resample(c1_raw, sampling_rate=sampling_rate, resample_number=resample_number, step=step, save=True, save_filepath=processed_data_directory + '/' + 'c1_1024_resample.npy')
c1_resample = np.load(r'data/processed/PHM2010/c1_1024_resample.npy')
c1_label = np.load(r'data/processed/PHM2010/c1_label.npy')
rs_rate = sampling_rate
c1_cwt = cwt(c1_resample, rs_rate, totalscale=totalscale, wavename=wavename, save=False)


exit()
