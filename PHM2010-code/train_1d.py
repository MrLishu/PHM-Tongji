import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
import torch
import torch.utils.data as Data
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, mean_absolute_error
from earlystopping import EarlyStopping
import torch.nn as nn
from model import WDCNN_512
from model import WDCNN_1024
from model import WDCNN_2048
from model import WDCNN_4096
from LSTM import LSTM
from tcn import TemporalConvNet
# from model215 import WDCNN_512
# from model215 import WDCNN_1024
# from model215 import WDCNN_2048
# from model215 import WDCNN_4096



EPOCH = 3000    #
BATCH_SIZE = 2048   #
# BATCH_SIZE = 1024   #
# BATCH_SIZE = 4096
LR = 0.005    #
# LR = 0.001
# num_para = 30
num_para = 1   #
name = '1C1024'

# 防止过拟合
patience = 50
early_stopping = EarlyStopping(patience)

# model = WDCNN_2048(C_in=6)
model = WDCNN_1024(C_in=6)   #
# model = LSTM()
# model = WDCNN_1024(C_in=6)

if torch.cuda.is_available():
    model = model.cuda()

# train_x1 = np.load(r'E:\data\1d_1C1024_data\1d_1C1024_data_c1.npy')  # [315 * num_para, length_para, 6]
# train_x2 = np.load(r'E:\data\1d_1C1024_data\1d_1C1024_data_c4.npy')
#
# train_y1 = np.load(r'E:\data\1d_1C1024_data\1d_1C1024_data_c1_labels.npy')
# train_y2 = np.load(r'E:\data\1d_1C1024_data\1d_1C1024_data_c4_labels.npy')
#
# test_x = np.load(r'E:\data\1d_1C1024_data\1d_1C1024_data_c6.npy')  # [315, length_para, 6]
# test_y = np.load(r'E:\data\1d_1C1024_data\1d_1C1024_data_c6_labels.npy')

#
# train_x1 = np.load(r'E:\data\1d_1C512_data\1d_1C512_data_c1.npy')  # [315 * num_para, length_para, 6]
# train_x2 = np.load(r'E:\data\1d_1C512_data\1d_1C512_data_c4.npy')
#
# train_y1 = np.load(r'E:\data\1d_1C512_data\1d_1C512_data_c1_labels.npy')
# train_y2 = np.load(r'E:\data\1d_1C512_data\1d_1C512_data_c4_labels.npy')
#
# test_x = np.load(r'E:\data\1d_1C512_data\1d_1C512_data_c6.npy')  # [315, length_para, 6]
# test_y = np.load(r'E:\data\1d_1C512_data\1d_1C512_data_c6_labels.npy')

# train_x1 = np.load(r'E:\data\1d_1C512_data\1d_1C512_data_c1.npy')  # [315 * num_para, length_para, 6]
# train_x2 = np.load(r'E:\data\1d_1C512_data\1d_1C512_data_c4.npy')  # [315 * num_para, length_para, 6]
train_x1 = np.load(r'E:\data\1d_1C1024_data\1d_1C1024_data_c1.npy')  # [315 * num_para, length_para, 6]
train_x2 = np.load(r'E:\data\1d_1C1024_data\1d_1C1024_data_c4.npy')  # [315 * num_para, length_para, 6]
# train_x1 = abs(train_x1)  # [315 * num_para, length_para, 6]
# train_x2 = abs(train_x2)
# train_x1 = np.log10(train_x1)
# train_x2 = np.log10(train_x2)
# train_x1 = train_x1[:, :, 0:1]
# train_x2 = train_x2[:, :, 0:1]

train_y1 = np.load(r'E:\data\1d_1C1024_data\1d_1C1024_data_c1_labels.npy')
train_y2 = np.load(r'E:\data\1d_1C1024_data\1d_1C1024_data_c4_labels.npy')

test_x = np.load(r'E:\data\1d_1C1024_data\1d_1C1024_data_c6.npy')  # [315, length_para, 6]
test_y = np.load(r'E:\data\1d_1C1024_data\1d_1C1024_data_c6_labels.npy')
# test_x = abs(test_x)
# test_x = np.log10(test_x)
# test_x = test_x[:, :, 0:1]
# train_x1 = train_x1[255:, :, 0:3]
# train_x2 = train_x2[255:, :, 0:3]
# test_x = test_x[255:, :, 0:3]
# train_y1 = train_y1[255:]
# train_y2 = train_y2[255:]
# test_y = test_y[255:]

# 前215
# train_x1 = train_x1[:215, :, :]
# train_x2 = train_x2[:215, :, :]
# test_x = test_x[:215, :, :]
# # train_x1 = train_x1[645:, :, :]
# # train_x2 = train_x2[645:, :, :]
# # test_x = test_x[645:, :, :]
# train_y1 = train_y1[:215]
# train_y2 = train_y2[:215]
# test_y = test_y[:215]

# 后100
train_x1 = train_x1[215:, :, :]
train_x2 = train_x2[215:, :, :]
test_x = test_x[215:, :, :]
# train_x1 = train_x1[645:, :, :]
# train_x2 = train_x2[645:, :, :]
# test_x = test_x[645:, :, :]
train_y1 = train_y1[215:]
train_y2 = train_y2[215:]
test_y = test_y[215:]


# # 转置
# train_x1 = train_x1.transpose((0, 2, 1))
# train_x2 = train_x2.transpose((0, 2, 1))
# test_x = test_x.transpose((0, 2, 1))

# train_x1 = np.load("E:\\data\\results\\1d-dataprocessing-results\\data_original_1_4096.npy")
# train_x2 = np.load("E:\\data\\results\\1d-dataprocessing-results\\data_original_4_4096.npy")
# test_x = np.load("E:\\data\\results\\1d-dataprocessing-results\\data_original_6_4096.npy")
# train_y1 = np.load("E:\\data\\results\\1d-dataprocessing-results\\data_y1.npy")
# train_y2 = np.load("E:\\data\\results\\1d-dataprocessing-results\\data_y4.npy")
# test_y = np.load("E:\\data\\results\\1d-dataprocessing-results\\data_y6.npy")


yy0 = test_y


def copy_labels(data, num_para):
    labels = np.empty(len(data) * num_para)
    for i in range(len(data)):
        for j in range(num_para):
            labels[i * num_para + j] = data[i]
    return labels


def fft(data):  # data shape: (n * length * channels)
    data_fft = np.empty([data.shape[0], data.shape[1], data.shape[2]])
    for i in range(data.shape[0]):
        for k in range(data.shape[2]):
            data_fft[i, :, k] = np.abs(np.fft.fft(data[i, :, k]))
    # print(data_fft.shape)
    return data_fft


mm = preprocessing.MinMaxScaler()   # 最小最大值标准化，将数据缩放至给定的最小值与最大值之间，通常是0与1之间，归一化？
train_y1 = mm.fit_transform(train_y1.reshape(-1, 1))   # fit_transform()先拟合数据，再标准化，即先拟合数据，然后转化它将其转化为标准形式
train_y2 = mm.fit_transform(train_y2.reshape(-1, 1))
test_y = mm.fit_transform(test_y.reshape(-1, 1))
test_y = test_y.reshape(-1)

train_x = np.append(train_x1, train_x2, axis=0)
train_y = np.append(train_y1, train_y2, axis=0)
# print(train_x.shape)
# print(train_y.shape)
# print(test_x.shape)
# print(test_y.shape)

# train_x = fft(train_x)
# test_x = fft(test_x)

train_y = copy_labels(train_y, num_para=num_para)
test_y = copy_labels(test_y, num_para=num_para)

print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)


# patience = 30
# early_stopping = EarlyStopping(patience)


# def train(model, train_x, train_y):
#     train_x = torch.from_numpy(train_x)
#     train_y = torch.from_numpy(train_y)
#
#     train_dataset = Data.TensorDataset(train_x, train_y)
#
#     all_num = train_x.shape[0]
#     train_num = int(all_num * 0.8)
#     train_data, val_data = Data.random_split(train_dataset, [train_num, all_num - train_num])
#
#     train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, )
#     val_loader = Data.DataLoader(dataset=val_data, batch_size=BATCH_SIZE, shuffle=True, )
#
#     optimizer = torch.optim.Adam(model.parameters(), lr=LR)
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)
#     loss_func = torch.nn.MSELoss()
#
#     train_loss = []
#     val_loss = []
#     # lr_list = []
#
#     for epoch in range(EPOCH):
#         total_loss = 0
#         total_loss2 = 0
#         model.train()
#         for step, (b_x, b_y) in enumerate(train_loader):
#             b_x = b_x.float()
#             b_y = b_y.float()
#             if torch.cuda.is_available():
#                 b_x = b_x.cuda()
#                 b_y = b_y.cuda()
#             output = model(b_x).squeeze(-1)
#             loss = loss_func(output, b_y)
#             L1_reg = 0
#             for param in model.parameters():
#                 L1_reg += torch.sum(torch.abs(param))
#             loss += 0.001 * L1_reg
#
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             scheduler.step()
#
#             total_loss += loss.cpu().item()
#
#         total_loss /= len(train_loader.dataset)
#         train_loss.append(total_loss)
#         # lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
#
#         model.eval()
#         with torch.no_grad():
#             for i, (v_x, v_y) in enumerate(val_loader):
#                 v_x = v_x.float()
#                 v_y = v_y.float()
#                 if torch.cuda.is_available():
#                     v_x = v_x.cuda()
#                     v_y = v_y.cuda()
#
#                 test_output = model(v_x).squeeze(-1)
#                 v_loss = loss_func(test_output, v_y)
#
#                 total_loss2 += v_loss.cpu().item()
#
#             total_loss2 /= len(val_loader.dataset)
#             val_loss.append(total_loss2)
#
#             # early_stopping(total_loss2, model)
#             # if early_stopping.early_stop:
#             #     print("Early stopping")
#             #     break
#
#         print('Train Epoch: {} \t Train Loss:{:.6f} \t Val Loss:{:.6f}'.format(epoch, total_loss, total_loss2))
#
#     X0 = np.array(train_loss).shape[0]
#     X1 = range(0, X0)
#     X2 = range(0, X0)
#     Y1 = train_loss
#     Y2 = val_loss
#     plt.subplot(2, 1, 1)
#     plt.plot(X1, Y1, '-')
#     plt.ylabel('train_loss')
#     plt.subplot(2, 1, 2)
#     plt.plot(X2, Y2, '-')
#     plt.ylabel('val_loss')
#     plt.show()
def train(model, train_x, train_y):
    train_x = torch.from_numpy(train_x)
    train_y = torch.from_numpy(train_y)

    train_dataset = Data.TensorDataset(train_x, train_y)

    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, )

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=800, gamma=0.9)
    loss_func = torch.nn.MSELoss()

    train_loss = []

    for epoch in range(EPOCH):
        total_loss = 0
        model.train()
        for step, (b_x, b_y) in enumerate(train_loader):
            b_x = b_x.float()
            b_y = b_y.float()
            if torch.cuda.is_available():
                b_x = b_x.cuda()
                b_y = b_y.cuda()
            output = model(b_x).squeeze(-1)
            loss = loss_func(output, b_y)
            L1_reg = 0
            for param in model.parameters():
                L1_reg += torch.sum(torch.abs(param))
            loss += 0.001 * L1_reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.cpu().item()

        total_loss /= len(train_loader.dataset)
        train_loss.append(total_loss)
        # lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])

        # early_stopping(total_loss2, model)
        # if early_stopping.early_stop:
        #     print("Early stopping")
        #     break

        # 防止过拟合
        if epoch > 200:
            early_stopping(total_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        print('Train Epoch: {} \t Train Loss:{:.6f}'.format(epoch, total_loss))

    X0 = np.array(train_loss).shape[0]
    X1 = range(0, X0)
    Y1 = train_loss
    plt.plot(X1, Y1, '-')
    plt.ylabel('train_loss')
    plt.show()


def tst(model, test_x, test_y):
    test_x = torch.from_numpy(test_x)
    test_y = torch.from_numpy(test_y)

    test_dataset = Data.TensorDataset(test_x, test_y)
    test_loader = Data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, )

    pred = torch.empty(1)

    model.eval()
    with torch.no_grad():
        for i, (tx, ty) in enumerate(test_loader):
            tx = tx.float()
            ty = ty.float()
            if torch.cuda.is_available():
                tx = tx.cuda()
                ty = ty.cuda()

            out = model(tx).squeeze(-1)
            pred = torch.cat((pred, out.cpu()))
            #  C = torch.cat( (A,B),0 )  #按维数0拼接（竖着拼）
            #  C = torch.cat( (A,B),1 )  #按维数1拼接（横着拼）

    pred = np.delete(pred.detach().numpy(), 0, axis=0)

    pred = mm.inverse_transform(pred.reshape(-1, 1))    # 进行数据归一化的还原

    pred = np.mean(pred.reshape(-1, num_para), axis=1)

    return pred


train(model, train_x, train_y)
# model.load_state_dict(torch.load('checkpoint.pt'))
pred = tst(model, test_x, test_y)

xx1 = range(0, yy0.shape[0])
yy1 = pred
print('yy1', yy1.shape)
print('yy1.reshape', yy1.reshape(-1).shape)
yy2 = signal.savgol_filter(pred.reshape(-1), 55, 5)


plt.plot(xx1, yy0, label='Actual Value')
plt.plot(xx1, yy1, 'red', label='Predicted Value')
plt.plot(xx1, yy2, 'k', label='Smoothed Pre-Value')

plt.xlabel('Times of cutting')
plt.ylabel(r'Average wear$\mu m$')
plt.legend(loc=4)
plt.show()

# 保存数据
# np.save(r'E:\data\results\npy\1c1024\data_' + str(name) + '_xx1.npy', xx1)
# 后100
np.save(r'E:\data\results\npy\1c1024\data_' + str(name) + 'last100_yy0.npy', yy0)
np.save(r'E:\data\results\npy\1c1024\data_' + str(name) + 'last100_yy1.npy', yy1)
np.save(r'E:\data\results\npy\1c1024\data_' + str(name) + 'last100_yy2.npy', yy2)
# 前215
# np.save(r'E:\data\results\npy\1c1024\data_' + str(name) + 'first200_yy0.npy', yy0)
# np.save(r'E:\data\results\npy\1c1024\data_' + str(name) + 'first200_yy1.npy', yy1)
# np.save(r'E:\data\results\npy\1c1024\data_' + str(name) + 'first200_yy2.npy', yy2)
# print(xx1)


mae = mean_absolute_error(yy0, yy1)
mae2 = mean_absolute_error(yy0, yy2)
rmse = math.sqrt(mean_squared_error(yy0, yy1))
rmse2 = math.sqrt(mean_squared_error(yy0, yy2))
print('Test MAE: %.3f' % mae)
print('MAE after Smoothing: %.3f' % mae2)
print('Test RMSE: %.3f' % rmse)
print('RMSE after Smoothing: %.3f' % rmse2)

