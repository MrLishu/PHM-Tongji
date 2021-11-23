import os
import scipy.signal as signal
import numpy as np
import pandas as pd
from sklearn import preprocessing
from scipy.fftpack import fft

length_para = 4096
# num_para = 30
# num_para = 1
num_para = 3

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
                # # 在第i个文件第k列的数据中，随机截取num_para段长度为length_para的数据
                # num = np.random.randint(low=10000, high=(len(data1) - 10000 - length_para))
                # data_train[i * num_para + j:, :, k] = data2[num:num + length_para]

                # 在第i个文件第k列的数据中，从正中间截取1段长度为length_para的数据  #
                start = (data2.shape[0] // 2) - (length_para // 2)
                end = (data2.shape[0] // 2) + (length_para // 2)
                data_train[i * num_para + j:, :, k] = data2[start: end]

                # # # 在第i个文件第k列的数据中，从1/4 1/2 3/4处截取3段长度为length_para的数据  #
                # start = ((j + 1) * data2.shape[0] // 4) - (length_para // 2)
                # end = ((j + 1) * data2.shape[0] // 4) + (length_para // 2)
                # data_train[i * num_para + j:, :, k] = data2[start: end]

                # # 在第i个文件第k列的数据中，隔点采样，得到长度为length_para的数据  #
                # data2 = data2[round(data2.shape[0] // 50) : - round(data2.shape[0] // 50)]
                # ber = round(data2.shape[0] // length_para)
                # for nu in range(length_para):
                #     data_train[i * num_para + j:, nu, k] = data2[ber * nu]

                # 进行FFT，获得频域数据
                data_train_fft = fft(data_train)

    print('data_train', data_train.shape)
    print('data_train_fft', data_train_fft.shape)

    # np.save(r'E:\data\1d_1C512'_data_' + str(name) + '_' + str(num_para) + '.npy', data_train)

    # 保存数据。在第i个文件第k列的数据中，从正中间截取1段长度为length_para的数据  或  从1/4 1/2 3/4处截取3段长度为length_para的数据
    np.save(r'E:\data\1d_' + str(num_para) + 'C' + str(length_para) + '_data_' + str(name) + '.npy', data_train)
    np.save(r'E:\data\1d_' + str(num_para) + 'C' + str(length_para) + '_data_' + str(name) + '_fft.npy', data_train_fft)

    # # 保存数据。在第i个文件第k列的数据中，隔点采样，得到长度为length_para的数据
    # np.save(r'E:\data\1d_intervals_' + str(length_para) + '_data_' + str(name) + '.npy', data_train)
    # np.save(r'E:\data\1d_intervals_' + str(length_para) + '_data_' + str(name) + '_fft.npy', data_train_fft)

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

    np.save(r'E:\data\1d_' + str(num_para) + 'C' + str(length_para) + '_data_' + str(name) + '_labels.npy', data)
    # np.save(r'E:\data\1d_intervals_' + str(length_para) + '_data_' + str(name) + '_labels.npy', data)

proprecessing(path=r'E:\data\c1', name='c1', length_para=length_para, num_para=num_para)
proprecessing(path=r'E:\data\c4', name='c4', length_para=length_para, num_para=num_para)
proprecessing(path=r'E:\data\c6', name='c6', length_para=length_para, num_para=num_para)

get_labels(path_with_f_name=r'E:\data\c1_wear.csv', name='c1')
get_labels(path_with_f_name=r'E:\data\c4_wear.csv', name='c4')
get_labels(path_with_f_name=r'E:\data\c6_wear.csv', name='c6')




