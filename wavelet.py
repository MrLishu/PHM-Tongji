import numpy as np
import matplotlib.pyplot as plt
import pywt

data_x4 = np.load(r"/Users/leo/Desktop/1d_1C4096_data_c4.npy")
k=data_x4[1]
print(k.shape)
num=k.T
data = num[5].T



sampling_rate = 4096#采样频率
t = np.arange(0,1.0,1.0/sampling_rate)  #0-1.0之间的数，步长为1.0/sampling_rate
f1 = 100#频率
f2 = 200
f3 = 300
wavename = "cgau8"  #小波函数
totalscal = 256     #totalscal是对信号进行小波变换时所用尺度序列的长度(通常需要预先设定好)
fc = pywt.central_frequency(wavename)#计算小波函数的中心频率
cparam = 2 * fc * totalscal  #常数c
scales = cparam/np.arange(totalscal,1,-1) #为使转换后的频率序列是一等差序列，尺度序列必须取为这一形式（也即小波尺度）
[cwtmatr, frequencies] = pywt.cwt(data,scales,wavename,1.0/sampling_rate)#连续小波变换模块

plt.figure(figsize=(8, 4))
plt.subplot(211) #第一整行
plt.plot(t, data)
plt.xlabel(u"time(s)")
plt.title(u"force point")
plt.subplot(212) #第二整行

plt.contourf(t, frequencies, abs(cwtmatr))
plt.ylabel(u"freq(Hz)")
plt.xlabel(u"time(s)")
plt.subplots_adjust(hspace=0.4) #调整边距和子图的间距 hspace为子图之间的空间保留的高度，平均轴高度的一部分
plt.show()




