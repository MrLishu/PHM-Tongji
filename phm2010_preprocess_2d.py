import os
import scipy.signal as signal
import numpy as np
import pandas as pd
from sklearn import preprocessing
import pywt
import scipy
from scipy.ndimage.interpolation import zoom
from scipy.fftpack import fft

length_para = 1024
num_para = 1

def cwt_demo(data, wavename, sampling_rate, totalscale):
    """
    :param data: shape [numbers, length, channels] [100, 2014, 1]
    :param wavename:
    :param sampling_rate: 5000
    :param totalscale: 256 or 512
    :return: cwtmatrix  shape [numbers, scale, length, channels] [100, 256, 1024, 1]
    """

    cwtmatrix = np.empty([data.shape[0], totalscale, data.shape[1], data.shape[2]])
    fc = pywt.central_frequency(wavename)
    cparam = 2 * fc * totalscale
    scales = cparam / np.arange(totalscale, 0, -1)
    for i in range(data.shape[2]):
        # for j in range(data.shape[0]):
        for j in range(data.shape[1]):      #################
            dataset = data[j, :, i]
            print('dataset', '[i, j]', [i, j])    ######################
            [cwtmatr, frequencies] = pywt.cwt(dataset, scales, wavename, 1.0 / sampling_rate)
            cwtmatrix[j, :, :, i] = cwtmatr
    return cwtmatrix   # [numbers, scale, length, channels] [100, 256, 1024, 1]

def cwt_process(data):
    dataset_new = []

    for i in range(data.shape[0]):
        dataset = data[i, :, :, :]  # 100*1024*1
        print('dataset', '[i]', [i])          #################
        data_cwt = cwt_demo(dataset, 'morl', 5000, 256)  # 100*256*1024*1
        print('data_cwt', data_cwt.shape)     #############

        dataset_new.append(data_cwt.transpose((0, 3, 1, 2)))

    dataset_new = np.array(dataset_new)
    dataset_new = data_resize(dataset_new, 224)   # 100*224*224*3
    print(dataset_new.shape)
    return dataset_new

def data_resize(data, size):
    s1, s2, s3, s4, s5 = data.shape
    data_rs = np.empty([s1, s2, s3, size, size])
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            for k in range(data.shape[2]):
                d = data[i, j, k, :, :]
                data_rs[i, j, k, :, :] = scipy.ndimage.zoom(d, (size / data.shape[3], size / data.shape[4]), order = 3)
    return data_rs

def proprecessing(path, name, length_para, num_para):
    files = os.listdir(path)
    files.sort(key=lambda x: int(x[4:-4]))
    s = []
    for file in files:
        if not os.path.isdir(path + file):
            f_name = str(file)
            tr = '\\'
            filename = path + tr + f_name
            s.append(filename)

    data_train = np.empty([315 * num_para, length_para, 6])

    for i in range(315):
        data1 = pd.read_csv(s[i], names=['Fx', 'Fy', 'Fz', 'Ax', 'Ay', 'Az', 'AE_rms'])
        for k in range(6):
            data2 = data1.iloc[:, k]

            scaler = preprocessing.StandardScaler()
            data2 = np.array(data2).reshape(-1, 1)
            data2 = scaler.fit_transform(data2.reshape(-1, 1))
            data2 = data2.reshape(-1)
            print(name, '[i, k]', [i, k])  #

            for j in range(num_para):
                # 在第i个文件第k列的数据中，从正中间截取1段长度为length_para的数据  #
                start = (data2.shape[0] // 2) - (length_para // 2)
                end = (data2.shape[0] // 2) + (length_para // 2)
                data_train[i * num_para + j:, :, k] = data2[start: end]


    print('data_train1', data_train.shape)
    data_train = cwt_process(data_train.reshape(data_train.shape[0], data_train.shape[1], data_train.shape[2], 1))
    print('data_train3', data_train.shape)

    np.save(r'data\2d_' + str(num_para) + 'C' + str(length_para) + '_data_' + str(name) + '.npy', data_train)


def get_labels(path_with_f_name, name):
    data0 = pd.read_csv(path_with_f_name)
    y1 = np.array(data0['flute_1'])
    y2 = np.array(data0['flute_2'])
    y3 = np.array(data0['flute_3'])
    y1 = y1.reshape(y1.shape[0], 1)
    y2 = y2.reshape(y2.shape[0], 1)
    y3 = y3.reshape(y3.shape[0], 1)
    y = np.concatenate((y1, y2, y3), axis=1)
    data = np.mean(y, 1)

    print(name, 'labels', data.shape)   #

    np.save(r'data\2d_' + str(num_para) + 'C' + str(length_para) + '_data_' + str(name) + '_labels.npy', data)
    # np.save(r'E:\data\1d_intervals_' + str(length_para) + '_data_' + str(name) + '_labels.npy', data)

proprecessing(path=r'data\c1', name='c1', length_para=length_para, num_para=num_para)
# proprecessing(path=r'E:\data\c4', name='c4', length_para=length_para, num_para=num_para)
# proprecessing(path=r'E:\data\c6', name='c6', length_para=length_para, num_para=num_para)
#
# get_labels(path_with_f_name=r'E:\data\c1_wear.csv', name='c1')
# get_labels(path_with_f_name=r'E:\data\c4_wear.csv', name='c4')
# get_labels(path_with_f_name=r'E:\data\c6_wear.csv', name='c6')




