import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
import os
import gan_sin

POINT = np.linspace(0, 10, 50)

# 创建保存图片位置
if not os.path.isdir("test_images"):
    os.mkdir("test_images")

device = torch.device("cpu")

class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.fc1 = nn.Linear(50, 128)
        self.fc2 = nn.Linear(128, 50)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# 加载模型
generator = torch.load('./save_Model/model_sin.pth')  # 导入网络的参数

g_noises = np.random.randn(64, 50)
g_noises = Variable(torch.Tensor(g_noises)).to(device)

fake_date = generator(g_noises)
# 为真实数据加上噪声
real_data = np.vstack([np.sin(POINT)])
real_data = Variable(torch.Tensor(real_data)).to(device)

plt.cla()
plt.plot(POINT, fake_date[0].to('cpu').detach().numpy(), c='#4AD631', lw=2, label="generated line")  # 生成网络生成的数据
plt.plot(POINT, real_data[0].to('cpu').detach().numpy(), c='#74BCFF', lw=3, label="real sin")  # 真实数据
plt.draw()

f = plt.gcf()  # 获取当前图像
plt.pause(5)
f.savefig('./test_images/test_sin.png')
