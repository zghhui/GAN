import argparse
import os
import numpy as np
import math
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
from tools.my_dataset import myDataset
import torch
#生成图片大小
img_shape = (3, 128,128)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *img_shape)
        return img

#创建保存图片位置
if not os.path.isdir("test_image"):
    os.mkdir("test_image")
#加载模型
generator=torch.load('generator.pth');
Tensor = torch.FloatTensor

# 噪声输入
z = Variable(Tensor(np.random.normal(0, 3, (2, 100))))

# 生成图像
gen_imgs = generator(z)
save_image(gen_imgs.data[:25], "test_image/test_7_16_14.png", nrow=5, normalize=True)