import os
import pandas as pd



def loadMillData(data_directory=r'data/raw/mill'):
    pass

def loadPHMData(data_directory=r'data/raw/PHM2010', dataset_names=['c1', 'c4', 'c6'], dataframe_number=-1):
    datasets = []

    for name in dataset_names:
        dataset_directory = os.path.join(data_directory, name)
        dataframe_number = len(os.listdir(dataset_directory)) if dataframe_number < 0 else dataframe_number
        signal_name = ['Fx', 'Fy', 'Fz', 'Ax', 'Ay', 'Az', 'AE_rms']
        sampling_rate = 50000

        raw_dataset = []
        for i in range(dataframe_number):
            df = pd.read_csv(os.path.join(dataset_directory, rf'c_{name[1]}_{i + 1:03d}.csv'), names=signal_name)
            raw_dataset.append(df)
            print(f'\rLoading dataset "{name}"... ({i + 1}/{dataframe_number})', end='')

        label_filepath = os.path.join(data_directory, rf'{name}_wear.csv')
        label_dataset = pd.read_csv((label_filepath)).to_numpy()

        datasets.append((raw_dataset, label_dataset))

    print(f'\nData loading completed.')
    return datasets



if __name__ == '__main__':
    loadPHMData('data/raw/PHM2010', ['c1'], 10)