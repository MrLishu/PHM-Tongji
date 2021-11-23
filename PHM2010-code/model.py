import torch.nn as nn

# 输入数据为4096
# 卷积层使用 torch.nn.Conv2d
# 激活层使用 torch.nn.ReLU
# 池化层使用 torch.nn.MaxPool2d
# 全连接层使用 torch.nn.Linear
# 归一化使用nn.BatchNorm2d(卷积的输出通道数),最常用于卷积网络中(防止梯度消失或爆炸)

# class WDCNN_4096(nn.Module):
#     def __init__(self, C_in):
#         super(WDCNN_4096, self).__init__()
#         # wdcnn
#         self.net = nn.Sequential(
#             nn.Conv1d(in_channels=C_in, out_channels=16, kernel_size=19, stride=20, padding=1),  # 4096-253   out_size = 向下取整[(input_size - kernel_size + (2 * padding)) / s ] + 1
#             nn.BatchNorm1d(16, affine=True),
#             nn.ReLU(inplace=True),
#             nn.MaxPool1d(kernel_size=2, stride=2),  # 默认padding=0, 253-126
#
#             nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),  # 126-126
#             nn.BatchNorm1d(32, affine=True),
#             nn.ReLU(inplace=True),
#             nn.MaxPool1d(kernel_size=2, stride=2),  # 126-63
#
#             nn.Conv1d(in_channels=32, out_channels=64, kernel_size=2, stride=1, padding=1),  # 63-64
#             nn.BatchNorm1d(64, affine=True),
#             nn.ReLU(inplace=True),
#             nn.MaxPool1d(kernel_size=2, stride=2),  # 64-32
#
#             nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),  # 32-32
#             nn.BatchNorm1d(64, affine=True),
#             nn.ReLU(inplace=True),
#             nn.MaxPool1d(kernel_size=2, stride=2),  # 32-16
#
#             nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),  # 16-14
#             nn.BatchNorm1d(64, affine=True),
#             nn.ReLU(inplace=True),
#             nn.MaxPool1d(kernel_size=2, stride=2),  # 14-7
#         )
#
#         # Dense layer  全连接层
#         self.fc = nn.Sequential(
#             nn.Linear(in_features=64*5, out_features=100),
#             nn.ReLU(),
#             nn.Linear(in_features=100, out_features=100),
#             nn.ReLU(),
#             nn.Linear(in_features=100, out_features=20)
#         )
#         self.classifier = nn.Sequential(
#             nn.Linear(in_features=20, out_features=1),
#             # nn.LogSoftmax(dim=1)
#         )
#
#     def forward(self, x):
#         x = x.permute(0, 2, 1).contiguous()
#         output = self.net(x)
#         output = output.view(output.size()[0], -1)
#         output = self.fc(output)
#         output = self.classifier(output)
#         return output


class WDCNN_4096(nn.Module):
    def __init__(self, C_in):
        super(WDCNN_4096, self).__init__()
        # wdcnn
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=C_in, out_channels=16, kernel_size=5, stride=16, padding=1),  # 4096-253   out_size = 向下取整[(input_size - kernel_size + (2 * padding)) / s ] + 1
            nn.BatchNorm1d(16, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 默认padding=0, 253-126

            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),  # 126-126
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 126-63

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=2, stride=1, padding=1),  # 63-64
            nn.BatchNorm1d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 64-32

            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),  # 32-32
            nn.BatchNorm1d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 32-16

            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),  # 16-14
            nn.BatchNorm1d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 14-7
        )

        # Dense layer  全连接层
        self.fc = nn.Sequential(
            nn.Linear(in_features=64*7, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=20)
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=20, out_features=1),
            # nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1).contiguous()
        output = self.net(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        output = self.classifier(output)
        return output



# 输入数据为512
# 卷积层使用 torch.nn.Conv2d
# 激活层使用 torch.nn.ReLU
# 池化层使用 torch.nn.MaxPool2d
# 全连接层使用 torch.nn.Linear
# 归一化使用nn.BatchNorm2d(卷积的输出通道数),最常用于卷积网络中(防止梯度消失或爆炸)

# class WDCNN_512(nn.Module):
#     def __init__(self, C_in):
#         super(WDCNN_512, self).__init__()
#         # wdcnn
#         self.net = nn.Sequential(
#             nn.Conv1d(in_channels=C_in, out_channels=16, kernel_size=64, stride=16, padding=1),  # 512-29   out_size = 向下取整[(input_size - kernel_size + (2 * padding)) / s ] + 1
#             nn.BatchNorm1d(16, affine=True),
#             nn.ReLU(inplace=True),
#             nn.MaxPool1d(kernel_size=2, stride=2),  # 默认padding=0, 29-14
#
#             nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),  # 14-14
#             nn.BatchNorm1d(32, affine=True),
#             nn.ReLU(inplace=True),
#             nn.MaxPool1d(kernel_size=2, stride=2),  # 14-7
#
#             # nn.Conv1d(in_channels=32, out_channels=64, kernel_size=2, stride=1, padding=1),  # 63-64
#             # nn.BatchNorm1d(64, affine=True),
#             # nn.ReLU(inplace=True),
#             # nn.MaxPool1d(kernel_size=2, stride=2),  # 64-32
#             #
#             # nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),  # 32-32
#             # nn.BatchNorm1d(64, affine=True),
#             # nn.ReLU(inplace=True),
#             # nn.MaxPool1d(kernel_size=2, stride=2),  # 32-16
#             #
#             # nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),  # 16-14
#             # nn.BatchNorm1d(64, affine=True),
#             # nn.ReLU(inplace=True),
#             # nn.MaxPool1d(kernel_size=2, stride=2),  # 14-7
#         )
#
#         # Dense layer  全连接层
#         self.fc = nn.Sequential(
#             # nn.Linear(in_features=64*7, out_features=100),
#             nn.Linear(in_features=32 * 7, out_features=100),
#             nn.ReLU(),
#             nn.Linear(in_features=100, out_features=100),
#             nn.ReLU(),
#             nn.Linear(in_features=100, out_features=20)
#         )
#         self.classifier = nn.Sequential(
#             nn.Linear(in_features=20, out_features=1),
#             # nn.LogSoftmax(dim=1)
#         )
#
#     def forward(self, x):
#         x = x.permute(0, 2, 1).contiguous()
#         output = self.net(x)
#         output = output.view(output.size()[0], -1)
#         output = self.fc(output)
#         output = self.classifier(output)
#         return output

class WDCNN_512(nn.Module):
    def __init__(self, C_in):
        super(WDCNN_512, self).__init__()
        # wdcnn
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=C_in, out_channels=16, kernel_size=3, stride=16, padding=1),  # 512-32   out_size = 向下取整[(input_size - kernel_size + (2 * padding)) / s ] + 1
            nn.BatchNorm1d(16, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 默认padding=0, 32-16

            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),  # 16-16
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 16-8

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),  # 8-8
            nn.BatchNorm1d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 8-4
            #
            # nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),  # 32-32
            # nn.BatchNorm1d(64, affine=True),
            # nn.ReLU(inplace=True),
            # nn.MaxPool1d(kernel_size=2, stride=2),  # 32-16
            #
            # nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),  # 16-14
            # nn.BatchNorm1d(64, affine=True),
            # nn.ReLU(inplace=True),
            # nn.MaxPool1d(kernel_size=2, stride=2),  # 14-7
        )

        # Dense layer  全连接层
        self.fc = nn.Sequential(
            # nn.Linear(in_features=64*7, out_features=100),
            nn.Linear(in_features=64 * 4, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=20)
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=20, out_features=1),
            # nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1).contiguous()
        output = self.net(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        output = self.classifier(output)
        return output



# class WDCNN_512(nn.Module):
#     def __init__(self, C_in):
#         super(WDCNN_512, self).__init__()
#         # wdcnn
#         self.net = nn.Sequential(
#             nn.Conv1d(in_channels=C_in, out_channels=16, kernel_size=19, stride=16, padding=1),  # 512-31   out_size = 向下取整[(input_size - kernel_size + (2 * padding)) / s ] + 1
#             nn.BatchNorm1d(16, affine=True),
#             nn.ReLU(inplace=True),
#             nn.MaxPool1d(kernel_size=2, stride=2),  # 默认padding=0, 31-15
#
#             nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),  # 15-15
#             nn.BatchNorm1d(32, affine=True),
#             nn.ReLU(inplace=True),
#             nn.MaxPool1d(kernel_size=2, stride=2),  # 15-7
#
#             # nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),  # 8-8
#             # nn.BatchNorm1d(64, affine=True),
#             # nn.ReLU(inplace=True),
#             # nn.MaxPool1d(kernel_size=2, stride=2),  # 8-4
#
#             # nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),  # 32-32
#             # nn.BatchNorm1d(64, affine=True),
#             # nn.ReLU(inplace=True),
#             # nn.MaxPool1d(kernel_size=2, stride=2),  # 9-4
#
#             # nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),  # 16-14
#             # nn.BatchNorm1d(64, affine=True),
#             # nn.ReLU(inplace=True),
#             # nn.MaxPool1d(kernel_size=2, stride=2),  # 14-7
#         )
#
#         # Dense layer  全连接层
#         self.fc = nn.Sequential(
#             # nn.Linear(in_features=64*7, out_features=100),
#             nn.Linear(in_features=32 * 7, out_features=100),
#             nn.ReLU(),
#             nn.Linear(in_features=100, out_features=100),
#             nn.ReLU(),
#             nn.Linear(in_features=100, out_features=20)
#         )
#         self.classifier = nn.Sequential(
#             nn.Linear(in_features=20, out_features=1),
#             # nn.LogSoftmax(dim=1)
#         )
#
#     def forward(self, x):
#         x = x.permute(0, 2, 1).contiguous()
#         output = self.net(x)
#         output = output.view(output.size()[0], -1)
#         output = self.fc(output)
#         output = self.classifier(output)
#         return output


# class WDCNN_512(nn.Module):
#     def __init__(self, C_in):
#         super(WDCNN_512, self).__init__()
#         # wdcnn
#         self.net = nn.Sequential(
#             nn.Conv1d(in_channels=C_in, out_channels=16, kernel_size=27, stride=16, padding=1),  # 512-31   out_size = 向下取整[(input_size - kernel_size + (2 * padding)) / s ] + 1
#             nn.BatchNorm1d(16, affine=True),
#             nn.ReLU(inplace=True),
#             nn.MaxPool1d(kernel_size=2, stride=2),  # 默认padding=0, 31-15
#
#             nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),  # 15-15
#             nn.BatchNorm1d(32, affine=True),
#             nn.ReLU(inplace=True),
#             nn.MaxPool1d(kernel_size=2, stride=2),  # 15-7
#
#             # nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),  # 8-8
#             # nn.BatchNorm1d(64, affine=True),
#             # nn.ReLU(inplace=True),
#             # nn.MaxPool1d(kernel_size=2, stride=2),  # 8-4
#
#             # nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),  # 32-32
#             # nn.BatchNorm1d(64, affine=True),
#             # nn.ReLU(inplace=True),
#             # nn.MaxPool1d(kernel_size=2, stride=2),  # 9-4
#
#             # nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),  # 16-14
#             # nn.BatchNorm1d(64, affine=True),
#             # nn.ReLU(inplace=True),
#             # nn.MaxPool1d(kernel_size=2, stride=2),  # 14-7
#         )
#
#         # Dense layer  全连接层
#         self.fc = nn.Sequential(
#             # nn.Linear(in_features=64*7, out_features=100),
#             nn.Linear(in_features=32 * 7, out_features=100),
#             nn.ReLU(),
#             nn.Linear(in_features=100, out_features=100),
#             nn.ReLU(),
#             nn.Linear(in_features=100, out_features=20)
#         )
#         self.classifier = nn.Sequential(
#             nn.Linear(in_features=20, out_features=1),
#             # nn.LogSoftmax(dim=1)
#         )
#
#     def forward(self, x):
#         x = x.permute(0, 2, 1).contiguous()
#         output = self.net(x)
#         output = output.view(output.size()[0], -1)
#         output = self.fc(output)
#         output = self.classifier(output)
#         return output


# class WDCNN_1024(nn.Module):
#     def __init__(self, C_in):
#         super(WDCNN_1024, self).__init__()
#         # wdcnn
#         self.net = nn.Sequential(
#             nn.Conv1d(in_channels=C_in, out_channels=16, kernel_size=27, stride=16, padding=1),  # 512-32   out_size = 向下取整[(input_size - kernel_size + (2 * padding)) / s ] + 1
#             nn.BatchNorm1d(16, affine=True),
#             nn.ReLU(inplace=True),
#             nn.MaxPool1d(kernel_size=2, stride=2),  # 默认padding=0, 32-16
#
#             nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),  # 16-16
#             nn.BatchNorm1d(32, affine=True),
#             nn.ReLU(inplace=True),
#             nn.MaxPool1d(kernel_size=2, stride=2),  # 16-8
#
#             nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),  # 8-8
#             nn.BatchNorm1d(64, affine=True),
#             nn.ReLU(inplace=True),
#             nn.MaxPool1d(kernel_size=2, stride=2),  # 8-4
#
#             # nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),  # 32-32
#             # nn.BatchNorm1d(64, affine=True),
#             # nn.ReLU(inplace=True),
#             # nn.MaxPool1d(kernel_size=2, stride=2),  # 32-16
#
#             # nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),  # 16-14
#             # nn.BatchNorm1d(64, affine=True),
#             # nn.ReLU(inplace=True),
#             # nn.MaxPool1d(kernel_size=2, stride=2),  # 14-7
#         )
#
#         # Dense layer  全连接层
#         self.fc = nn.Sequential(
#             # nn.Linear(in_features=64*7, out_features=100),
#             nn.Linear(in_features=64 * 7, out_features=100),
#             nn.ReLU(),
#             nn.Linear(in_features=100, out_features=100),
#             nn.ReLU(),
#             nn.Linear(in_features=100, out_features=20)
#         )
#         self.classifier = nn.Sequential(
#             nn.Linear(in_features=20, out_features=1),
#             # nn.LogSoftmax(dim=1)
#         )
#
#     def forward(self, x):
#         x = x.permute(0, 2, 1).contiguous()
#         output = self.net(x)
#         output = output.view(output.size()[0], -1)
#         output = self.fc(output)
#         output = self.classifier(output)
#         return output



class WDCNN_1024(nn.Module):
    def __init__(self, C_in):
        super(WDCNN_1024, self).__init__()
        # wdcnn
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=C_in, out_channels=16, kernel_size=3, stride=16, padding=1),  # 1024-63  out_size = 向下取整[(input_size - kernel_size + (2 * padding)) / s ] + 1
            nn.BatchNorm1d(16, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 默认padding=0, 63-31

            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),  # 31-31
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 31-15

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),  # 15-16
            nn.BatchNorm1d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 16-8

           # nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),  # 8-8
           # nn.BatchNorm1d(64, affine=True),
           # nn.ReLU(inplace=True),
           # nn.MaxPool1d(kernel_size=2, stride=2),  # 8-4

            # nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),  # 16-14
            # nn.BatchNorm1d(64, affine=True),
            # nn.ReLU(inplace=True),
            # nn.MaxPool1d(kernel_size=2, stride=2),  # 14-7
        )

        # Dense layer  全连接层
        self.fc = nn.Sequential(
            # nn.Linear(in_features=64*7, out_features=100),
            nn.Linear(in_features=64 * 8, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=20)
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=20, out_features=1),
            # nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1).contiguous()
        output = self.net(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        output = self.classifier(output)
        return output




class WDCNN_2048(nn.Module):
    def __init__(self, C_in):
        super(WDCNN_2048, self).__init__()
        # wdcnn
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=C_in, out_channels=16, kernel_size=64, stride=16, padding=1),  # 2048-125   out_size = 向下取整[(input_size - kernel_size + (2 * padding)) / s ] + 1
            nn.BatchNorm1d(16, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 默认padding=0, 125-62

            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),  # 62-62
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 62-31

             nn.Conv1d(in_channels=32, out_channels=64, kernel_size=2, stride=1, padding=1),  # 31-32
             nn.BatchNorm1d(64, affine=True),
             nn.ReLU(inplace=True),
             nn.MaxPool1d(kernel_size=2, stride=2),  # 32-16

             nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),  # 16-16
             nn.BatchNorm1d(64, affine=True),
             nn.ReLU(inplace=True),
             nn.MaxPool1d(kernel_size=2, stride=2),  # 16-8

            # nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),  # 16-14
            # nn.BatchNorm1d(64, affine=True),
            # nn.ReLU(inplace=True),
            # nn.MaxPool1d(kernel_size=2, stride=2),  # 14-7
        )

        # Dense layer  全连接层
        self.fc = nn.Sequential(
            # nn.Linear(in_features=64*7, out_features=100),
            nn.Linear(in_features=64 * 8, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=20)
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=20, out_features=1),
            # nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1).contiguous()
        output = self.net(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        output = self.classifier(output)
        return output
