import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt

SAMPLE_GAP = 0.2
SAMPLE_NUM = 50
N_GNET = 50
BATCH_SIZE = 64
USE_CUDA = True
MAX_EPOCH = 1000000
POINT = np.linspace(0, SAMPLE_GAP * SAMPLE_NUM, SAMPLE_NUM)


# 判别器
class disciminator(nn.Module):
    def __init__(self):
        super(disciminator, self).__init__()
        self.fc1 = nn.Linear(SAMPLE_NUM, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.squeeze(-1)
        return F.sigmoid(x)


# 生成器
class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.fc1 = nn.Linear(N_GNET, 128)
        self.fc2 = nn.Linear(128, SAMPLE_NUM)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

PATH = r"D:\Code\Program\Python_Code\GAN\gan_sin\Model"
def main():
    plt.ion()  # 开启interactive mode，便于连续plot
    # 用于计算的设备 CPU or GPU
    device = torch.device("cpu")
    # 定义判别器与生成器的网络
    net_d = disciminator()
    net_g = generator()
    net_d.to(device)
    net_g.to(device)
    # 损失函数
    criterion = nn.BCELoss().to(device)
    # 真假数据的标签
    true_lable = Variable(torch.ones(BATCH_SIZE)).to(device)
    fake_lable = Variable(torch.zeros(BATCH_SIZE)).to(device)
    # 优化器
    optimizer_d = torch.optim.Adam(net_d.parameters(), lr=0.0001)
    optimizer_g = torch.optim.Adam(net_g.parameters(), lr=0.0001)
    for i in range(MAX_EPOCH):
        # 为真实数据加上噪声
        real_data = np.vstack([np.sin(POINT) + np.random.normal(0, 0.01, SAMPLE_NUM) for _ in range(BATCH_SIZE)])
        real_data = Variable(torch.Tensor(real_data)).to(device)
        # 用随机噪声作为生成器的输入
        g_noises = np.random.randn(BATCH_SIZE, N_GNET)
        g_noises = Variable(torch.Tensor(g_noises)).to(device)
        # 训练判别器
        optimizer_d.zero_grad()
        # 辨别器辨别真图的loss
        d_real = net_d(real_data)
        loss_d_real = criterion(d_real, true_lable)
        loss_d_real.backward()
        # 辨别器辨别假图的loss
        fake_date = net_g(g_noises)
        d_fake = net_d(fake_date)
        loss_d_fake = criterion(d_fake, fake_lable)
        loss_d_fake.backward()
        optimizer_d.step()
        # 训练生成器

        optimizer_g.zero_grad()
        fake_date = net_g(g_noises)
        d_fake = net_d(fake_date)
        # 生成器生成假图的loss
        loss_g = criterion(d_fake, true_lable)
        loss_g.backward()
        optimizer_g.step()
        # 每200步画出生成的数字图片和相关的数据
        #
        if i % 10000 == 0:
            print(fake_date[0])
            plt.cla()
            plt.plot(POINT, fake_date[0].to('cpu').detach().numpy(), c='#4AD631', lw=2,
                     label="generated line")  # 生成网络生成的数据
            plt.plot(POINT, real_data[0].to('cpu').detach().numpy(), c='#74BCFF', lw=3, label="real sin")  # 真实数据
            prob = (loss_d_real.mean() + 1 - loss_d_fake.mean()) / 2.

            plt.draw()
            plt.pause(0.2)
            f = plt.gcf()  # 获取当前图像
            f.savefig(r'D:\Code\Program\Python_Code\GAN\gan_sin\train_images\train_%d.png'%(i/10000))
            f.clear()  # 释放内存

    #保存模型
    torch.save(net_g,'D:\Code\Program\Python_Code\GAN\gan_sin\save_Model\model_sin.pth' )

    plt.ioff()
    plt.show()


if __name__ == '__main__':
    main()