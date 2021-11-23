import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, mean_absolute_error
import scipy.signal as signal

EPOCH = 300  #
BATCH_SIZE = 1024  #
LR = 0.002  #
num_para = 1  #
bidirectional = True   #

# 提取每次走刀全部数据的数据特征及标签
# data_x1 = np.load("E:\\data\\dataprocessing-results\\data_x1.npy")
# data_x4 = np.load("E:\\data\\dataprocessing-results\\data_x4.npy")
# data_x6 = np.load("E:\\data\\dataprocessing-results\\data_x6.npy")
# data_y1 = np.load("E:\\data\\dataprocessing-results\\data_y1.npy")
# data_y4 = np.load("E:\\data\\dataprocessing-results\\data_y4.npy")
# data_y6 = np.load("E:\\data\\dataprocessing-results\\data_y6.npy")

# # 提取截去每次走刀左右两端各4096个数据的数据特征及标签
# data_x1 = np.load(r"E:\data\1d_1C512_data\1d_1C512_data_c1.npy")
# data_x4 = np.load(r"E:\data\1d_1C512_data\1d_1C512_data_c4.npy")
# data_x6 = np.load(r"E:\data\1d_1C512_data\1d_1C512_data_c6.npy")
# data_y1 = np.load(r"E:\data\1d_1C512_data\1d_1C512_data_c1_labels.npy")
# data_y4 = np.load(r"E:\data\1d_1C512_data\1d_1C512_data_c4_labels.npy")
# data_y6 = np.load(r"E:\data\1d_1C512_data\1d_1C512_data_c6_labels.npy")

data_x1 = np.load(r'E:\data\data_100_1d\data_c1_100_1d.npy')
data_x4 = np.load(r'E:\data\data_100_1d\data_c4_100_1d.npy')
data_x6 = np.load(r'E:\data\data_100_1d\data_c6_100_1d.npy')
data_y1 = np.load(r'E:\data\data_100_1d\data_c1_labels.npy')
data_y4 = np.load(r'E:\data\data_100_1d\data_c4_labels.npy')
data_y6 = np.load(r'E:\data\data_100_1d\data_c6_labels.npy')

# 取单通道
data_x1 = data_x1[:, :, 0:1]
data_x4 = data_x4[:, :, 0:1]
data_x6 = data_x6[:, :, 0:1]

# 转置
data_x1 = data_x1.transpose((0, 2, 1))
data_x4 = data_x4.transpose((0, 2, 1))
data_x6 = data_x6.transpose((0, 2, 1))

yy0 = data_y4


# data_x1 = np.load("E:\\data\\results\\1d-dataprocessing-results\\data_original_1_4096.npy")
# data_x4 = np.load("E:\\data\\results\\1d-dataprocessing-results\\data_original_4_4096.npy")
# data_x6 = np.load("E:\\data\\results\\1d-dataprocessing-results\\data_original_6_4096.npy")
# data_y1 = np.load("E:\\data\\results\\1d-dataprocessing-results\\data_y1.npy")
# data_y4 = np.load("E:\\data\\results\\1d-dataprocessing-results\\data_y4.npy")
# data_y6 = np.load("E:\\data\\results\\1d-dataprocessing-results\\data_y6.npy")


# data_x1 = np.load(r'E:\data\1d_1C512_data\1d_1C512_data_c1.npy')  # [315 * num_para, length_para, 6]
# train_x2 = np.load(r'E:\data\1d_1C512_data\1d_1C512_data_c4.npy')
#
# train_y1 = np.load(r'E:\data\1d_1C512_data\1d_1C512_data_c1_labels.npy')
# train_y2 = np.load(r'E:\data\1d_1C512_data\1d_1C512_data_c4_labels.npy')
#
# test_x = np.load(r'E:\data\1d_1C512_data\1d_1C512_data_c6.npy')  # [315, length_para, 6]
# test_y = np.load(r'E:\data\1d_1C512_data\1d_1C512_data_c6_labels.npy')



# def normalization(data):
#     mu = np.mean(data, axis=0)
#     sigma = np.std(data, axis=0)
#     return (data - mu) / sigma


# def norm_all(data):
#     d = np.empty((data.shape[0], data.shape[1], data.shape[2]))
#     for i in range(data.shape[1]):
#         data1 = data[:, i, :]
#         for j in range(data1.shape[0]):
#             data2 = data1[j, :]
#             d[j, i, :] = normalization(data2)
#     return d
#
#
# def normal_label(data):
#     minVals = data.min(0)
#     maxVals = data.max(0)
#     ranges = maxVals - minVals
#     normData = np.zeros(np.shape(data))
#     m = data.shape[0]
#     normData = data - np.tile(minVals, (m, 1))
#     normData = normData / np.tile(ranges, (m, 1))
#     return normData[0]
#
#
# data_x1 = norm_all(data_x1)
# data_x4 = norm_all(data_x4)
# data_x6 = norm_all(data_x6)
# data_y1 = normal_label(data_y1)
# data_y4 = normal_label(data_y4)
# data_y6 = normal_label(data_y6)

def copy_labels(data, num_para):
    labels = np.empty(len(data) * num_para)
    for i in range(len(data)):
        for j in range(num_para):
            labels[i * num_para + j] = data[i]
    return labels


# ##########  最小最大值标准化
mm = preprocessing.MinMaxScaler()   # 最小最大值标准化，将数据缩放至给定的最小值与最大值之间，通常是0与1之间，归一化？
data_y1 = mm.fit_transform(data_y1.reshape(-1, 1))
data_y4 = mm.fit_transform(data_y4.reshape(-1, 1))
data_y6 = mm.fit_transform(data_y6.reshape(-1, 1))
print('data_y1', data_y1.shape)
print('data_y4', data_y4.shape)
print('data_y6', data_y6.shape)
data_y1 = data_y1.reshape(-1)
data_y4 = data_y4.reshape(-1)
data_y6 = data_y6.reshape(-1)
print('data_y4-1', data_y4.shape)
# #########

train_x = np.append(data_x1, data_x6, axis=0)
train_y = np.append(data_y1, data_y6, axis=0)
test_x = data_x4
test_y = data_y4

print('train_x', train_x.shape)
print('test_x', test_x.shape)

train_y = copy_labels(train_y, num_para=num_para)
test_y = copy_labels(test_y, num_para=num_para)

# #####
# train_x = train_x.transpose((0, 2, 1))
# test_x = test_x.transpose((0, 2, 1))

train_x = torch.from_numpy(train_x)
train_y = torch.from_numpy(train_y)
test_x = torch.from_numpy(test_x)
test_y = torch.from_numpy(test_y)
print('train_x', train_x.shape, 'train_y', train_y.shape)
print('test_x', test_x.shape, 'test_y', test_y.shape)


train_dataset = Data.TensorDataset(train_x, train_y)
all_num = train_x.shape[0]
# train_num = int(all_num * 0.8)
train_num = int(all_num * 0.99)
train_data, val_data = Data.random_split(train_dataset, [train_num, all_num - train_num])
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, )
val_loader = Data.DataLoader(dataset=val_data, batch_size=BATCH_SIZE, shuffle=True, )

test_dataset = Data.TensorDataset(test_x, test_y)
test_loader = Data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, )

# # 单向LSTM
# class LSTM(nn.Module):
#     def __init__(self):
#         super(LSTM, self).__init__()
#         self.lstm = nn.LSTM(
#             input_size=6,
#             hidden_size=64,
#             num_layers=2,
#             batch_first=True,
#             dropout=0.05,
#         )
#         self.out = nn.Sequential(
#             nn.Linear(64, 10),
#             nn.BatchNorm1d(10, momentum=0.5),
#             nn.ReLU(),
#             nn.Linear(10, 1),
#         )
#
#     def forward(self, x):
#         x = x.permute(0, 2, 1).contiguous()
#         r_out, (h_n, h_c) = self.lstm(x, None)
#         out = self.out(r_out[:, -1, :])
#         return out


# 双向LSTM
# class LSTM(nn.Module):
#     def __init__(self):
#         super(LSTM, self).__init__()
#         self.num_layers=1
#         if bidirectional:
#             self.num_layers = 2
#         self.lstm = nn.LSTM(
#             input_size=6,   # x的特征维度
#             hidden_size=64,    # 隐藏层的特征维度
#             num_layers = self.num_layers,    # lstm隐层的层数，默认为1
#             dropout=0.05,     # 除最后一层，每一层的输出都进行dropout，默认为: 0
#             batch_first=True,   # True则输入输出的数据格式为 (batch, seq, feature)
#             bidirectional=bidirectional    # True则为双向lstm默认为False
#         )
#         # if bidirectional:
#         #     self.layer_size = 128
#         self.out = nn.Sequential(
#             nn.Linear(64 * self.num_layers, 10),
#             nn.BatchNorm1d(10, momentum=0.5),
#             nn.ReLU(),
#             nn.Linear(10, 1),
#         )
#     def forward(self, x):
#         x = x.permute(0, 2, 1).contiguous()
#         r_out, (h_n, h_c) = self.lstm(x, None)
#
#         # print("\n\n")
#         # print(r_out.shape)
#         out = self.out(r_out[:, -1, :])
#         # output=output.squeeze(-1)
#         return out


# 双向LSTM
class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.num_layers = 1
        if bidirectional:
            self.num_layers = 2
        self.lstm = nn.LSTM(
            input_size=1,   # x的特征维度
            # hidden_size=64,    # 隐藏层的特征维度
            hidden_size=128,  # 隐藏层的特征维度
            num_layers=self.num_layers,    # lstm隐层的层数，默认为1
            dropout=0.05,     # 除最后一层，每一层的输出都进行dropout，默认为: 0
            batch_first=True,   # True则输入输出的数据格式为 (batch, seq, feature)
            bidirectional=bidirectional    # True则为双向；lstm默认为False
        )
        # self.linear = nn.Linear(self.num_layers * 2 * 64, 1)

        # if bidirectional:H
        #     self.layer_size = 128
        self.out = nn.Sequential(
            # nn.Linear(64 * self.num_layers, 10),
            # nn.BatchNorm1d(10, momentum=0.5),
            # nn.ReLU(),
            # nn.Linear(10, 1),
            nn.Linear(128 * self.num_layers, 64),
            nn.BatchNorm1d(64, momentum=0.5),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32, momentum=0.5),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1).contiguous()

        r_out, (h_n, h_c) = self.lstm(x, None)

        # print("\n\n")
        # print(r_out.shape)
        out = self.out(r_out[:, -1, :])
        # output=output.squeeze(-1)
        return out



model = LSTM()
if torch.cuda.is_available():
    model = model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)
loss_func = torch.nn.MSELoss()

train_loss = []
val_loss = []
lr_list = []

for epoch in range(EPOCH):
    total_loss = 0
    total_loss2 = 0
    model.train()
    for step, (b_x, b_y) in enumerate(train_loader):
        b_x = b_x.float()
        b_y = b_y.float()
        if torch.cuda.is_available():
            b_x = b_x.cuda()
            b_y = b_y.cuda()
        output = model(b_x).squeeze(-1)
        loss = loss_func(output, b_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # total_loss += loss.cpu().item()
        total_loss += loss.cpu().item()

    total_loss /= len(train_loader.dataset)
    train_loss.append(total_loss)
    lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])

    model.eval()
    with torch.no_grad():
        for i, (v_x, v_y) in enumerate(val_loader):
            v_x = v_x.float()
            v_y = v_y.float()
            if torch.cuda.is_available():
                v_x = v_x.cuda()
                v_y = v_y.cuda()

            test_output = model(v_x).squeeze(-1)
            v_loss = loss_func(test_output, v_y)

            total_loss2 += v_loss.cpu().item()

        total_loss2 /= len(val_loader.dataset)
        val_loss.append(total_loss2)

    print('Train Epoch: {} \t Train Loss:{:.6f} \t Val Loss:{:.6f}'.format(epoch, total_loss, total_loss2))


X0 = np.array(train_loss).shape[0]
X1 = range(0, X0)
Y1 = train_loss
plt.plot(X1, Y1, '-')
plt.ylabel('train_loss')
plt.show()


# X0 = np.array(train_loss).shape[0]
# x1 = range(0, X0)
# x2 = range(0, X0)
# y1 = train_loss
# y2 = val_loss
# plt.subplot(2, 1, 1)
# plt.plot(x1, y1, '-')
# plt.ylabel('train_loss')
# plt.subplot(2, 1, 2)
# plt.plot(x2, y2, '-')
# plt.ylabel('val_loss')
# plt.show()

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

pred = np.delete(pred.detach().numpy(), 0, axis=0)

# 将归一化的数据还原
# mm = preprocessing.MinMaxScaler()
pred = mm.inverse_transform(pred.reshape(-1, 1))  # 进行数据归一化的还原

pred = np.mean(pred.reshape(-1, num_para), axis=1)

# xx1 = range(0, 315)
# yy1 = pred
# yy2 = test_y.cpu().detach().numpy()
# plt.plot(xx1, yy1, color='black', label='Predicted value')
# plt.plot(xx1, yy2, color='red', label='Actual value')
# plt.xlabel('Times of cutting')
# plt.ylabel(r'Average wear$\mu m$')
# plt.legend(loc=4)
# plt.show()
#
# rmse = math.sqrt(mean_squared_error(pred, yy2))
# print('Test RMSE: %.3f' % rmse)

xx1 = range(0, yy0.shape[0])
yy1 = pred
yy2 = signal.savgol_filter(pred.reshape(-1), 55, 5)

plt.plot(xx1, yy0, label='Actual Value')
plt.plot(xx1, yy1, 'red', label='Predicted Value')
plt.plot(xx1, yy2, 'k', label='Smoothed Pre-Value')

plt.xlabel('Times of cutting')
plt.ylabel(r'Average wear$\mu m$')
plt.legend(loc=4)
plt.show()

mae = mean_absolute_error(yy0, yy1)
mae2 = mean_absolute_error(yy0, yy2)
rmse = math.sqrt(mean_squared_error(yy0, yy1))
rmse2 = math.sqrt(mean_squared_error(yy0, yy2))
print('Test MAE: %.3f' % mae)
print('MAE after Smoothing: %.3f' % mae2)
print('Test RMSE: %.3f' % rmse)
print('RMSE after Smoothing: %.3f' % rmse2)

