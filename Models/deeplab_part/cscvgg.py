# -*- encoding: utf-8 -*-
# Author  : haitong
# Time    : 2024-01-25 16:39
# File    : cscvgg.py
# Software: PyCharm

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np





class VGGBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class CSCBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, T):
        super(CSCBlock, self).__init__()
        self.T = T
        # Convolutional Filters
        self.W1 = nn.Parameter(torch.randn(mid_channels, in_channels, 3, 3), requires_grad=True)
        self.W2 = nn.Parameter(torch.randn(out_channels, mid_channels, 3, 3), requires_grad=True)
        self.c1 = nn.Parameter(torch.ones(1, 1, 1, 1), requires_grad=True)
        self.c2 = nn.Parameter(torch.ones(1, 1, 1, 1), requires_grad=True)
        # Biases / Thresholds
        self.b1 = nn.Parameter(torch.zeros(1, mid_channels, 1, 1), requires_grad=True)
        self.b2 = nn.Parameter(torch.zeros(1, out_channels, 1, 1), requires_grad=True)
        # Initialization    # 2021年6月27日18:42:56
        self.W1.data = .1 / np.sqrt(in_channels * 9) * self.W1.data
        self.W2.data = .1 / np.sqrt(mid_channels * 9) * self.W2.data
        # BN                # 2021年6月27日18:43:02
        self.BN1 = nn.BatchNorm2d(mid_channels)
        self.BN2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # Encoding
        gamma1 = F.relu(self.BN1(self.c1 * F.conv2d(x, self.W1, stride=1, padding=1) + self.b1))
        gamma2 = F.relu(self.BN2(self.c2 * F.conv2d(gamma1, self.W2, stride=1, padding=1) + self.b2))

        for _ in range(self.T):
            #  backward computation
            gamma1 = F.conv_transpose2d(gamma2, self.W2, stride=1, padding=1)
            gamma1 = F.relu((gamma1 - self.c1 * F.conv2d(F.conv_transpose2d(gamma1, self.W1, stride=1, padding=1) - x,
                                                         self.W1, stride=1, padding=1)) + self.b1)
            gamma2 = F.relu((gamma2 - self.c2 * F.conv2d(
                F.conv_transpose2d(gamma2, self.W2, stride=1, padding=1) - gamma1, self.W2, stride=1,
                padding=1)) + self.b2)
        return gamma2


class CSCVGG(nn.Module):
    def __init__(self, unfolding):
        super(CSCVGG, self).__init__()

        # Encode
        self.conv_encode1 = CSCBlock(3, 64, 64, unfolding)
        self.conv_pool1 = nn.MaxPool2d(2, 2)
        self.conv_encode2 = CSCBlock(64, 128, 128, unfolding)
        self.conv_pool2 = nn.MaxPool2d(2, 2)
        self.conv_encode3 = CSCBlock(128, 256, 256, unfolding)
        self.conv_pool3 = nn.MaxPool2d(2, 2)
        self.conv_encode4 = CSCBlock(256, 512, 512, unfolding)
        self.conv_pool4 = nn.MaxPool2d(2, 2)

        # Bottleneck
        self.conv_encode5 = CSCBlock(512, 1024, 1024, unfolding)

    def forward(self, x):  # [1 3 512 512]
        encode_block1 = self.conv_encode1(x)  # [1 64 512 512]

        encode_pool1 = self.conv_pool1(encode_block1)  # [1 64 256 256]
        encode_block2 = self.conv_encode2(encode_pool1)  # [1 128 256 256]

        encode_pool2 = self.conv_pool2(encode_block2)  # [1 128 128 128]
        encode_block3 = self.conv_encode3(encode_pool2)  # [1 256 128 128]

        encode_pool3 = self.conv_pool3(encode_block3)  # [1 256 64 64]
        encode_block4 = self.conv_encode4(encode_pool3)  # [1 512 64 64]

        encode_pool4 = self.conv_pool4(encode_block4)  # [1 512 32 32]
        encode_block5 = self.conv_encode5(encode_pool4)  # [1 1024 32 32]

        return encode_block5, encode_block3



class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()

        # Encode
        self.conv_encode1 = VGGBlock(3, 64, 64)
        self.conv_pool1 = nn.MaxPool2d(2, 2)
        self.conv_encode2 = VGGBlock(64, 128, 128)
        self.conv_pool2 = nn.MaxPool2d(2, 2)
        self.conv_encode3 = VGGBlock(128, 256, 256)
        self.conv_pool3 = nn.MaxPool2d(2, 2)
        self.conv_encode4 = VGGBlock(256, 512, 512)
        self.conv_pool4 = nn.MaxPool2d(2, 2)

        # Bottleneck
        self.conv_encode5 = VGGBlock(512, 1024, 1024)

    def forward(self, x):  # [1 3 512 512]
        encode_block1 = self.conv_encode1(x)  # [1 64 512 512]

        encode_pool1 = self.conv_pool1(encode_block1)  # [1 64 256 256]
        encode_block2 = self.conv_encode2(encode_pool1)  # [1 128 256 256]

        encode_pool2 = self.conv_pool2(encode_block2)  # [1 128 128 128]
        encode_block3 = self.conv_encode3(encode_pool2)  # [1 256 128 128]

        encode_pool3 = self.conv_pool3(encode_block3)  # [1 256 64 64]
        encode_block4 = self.conv_encode4(encode_pool3)  # [1 512 64 64]

        encode_pool4 = self.conv_pool4(encode_block4)  # [1 512 32 32]
        encode_block5 = self.conv_encode5(encode_pool4)  # [1 1024 32 32]

        return encode_block5, encode_block3


if __name__ == "__main__":
    import torch

    model = CSCVGG(unfolding=1)
    # model = VGG()
    input = torch.rand(1, 3, 352, 480)

    output, low_level_feat = model(input)  # 1/16 1/4
    print(output.size())
    print(low_level_feat.size())
