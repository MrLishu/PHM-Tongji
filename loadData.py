import os
import pandas as pd



def loadMillData(data_directory='data/raw/mill'):
    pass

def loadPHMData(data_directory='data/raw/PHM2010', datasets=['c1', 'c4', 'c6'], dataframe_number=-1):
    raw_datasets = []
    label_datasets = []

    for name in datasets:
        dataset_directory = os.path.join(data_directory, name)
        dataframe_number = len(os.listdir(dataset_directory)) if dataframe_number < 0 else dataframe_number
        signal_name = ['Fx', 'Fy', 'Fz', 'Ax', 'Ay', 'Az', 'AE_rms']
        sampling_rate = 50000

        raw_datasets.append([])
        for i in range(dataframe_number):
            df = pd.read_csv(os.path.join(dataset_directory, f'c_{name[1]}_{i + 1:03d}.csv'), names=signal_name)
            raw_datasets[-1].append(df)
            print(f'\rLoading dataset "{name}"... ({i + 1}/{dataframe_number})', end='')

        label_filepath = os.path.join(data_directory, f'{name}_wear.csv')
        label_datasets.append(pd.read_csv((label_filepath)).to_numpy())

    print(f'\nData loading completed.')
    return raw_datasets, label_datasets



if __name__ == '__main__':
    loadPHMData('data/raw/PHM2010', ['c1'], 10)