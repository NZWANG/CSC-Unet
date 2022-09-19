# -*- encoding: utf-8 -*-
# Author  : Haitong

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class ML_ISTA_block(nn.Module):

    def __init__(self, T, in_channels, mid_channels, out_channels):
        super(ML_ISTA_block, self).__init__()
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


def final_block(in_channels, out_channels):
    block = nn.Conv2d(kernel_size=(1, 1), in_channels=in_channels, out_channels=out_channels)
    return block


class expansive_block(nn.Module):

    def __init__(self, T, in_channels, mid_channels, out_channels):
        super(expansive_block, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=(3, 3), stride=2, padding=1,
                                     output_padding=1, dilation=1)
        self.block = ML_ISTA_block(T=T, in_channels=in_channels, mid_channels=mid_channels, out_channels=out_channels)

    def forward(self, e, d):
        d = self.up(d)
        cat = torch.cat([e, d], dim=1)
        out = self.block(cat)
        return out


class CSC_UNet(nn.Module):
    def __init__(self, out_channel, unfolding):
        super(CSC_UNet, self).__init__()
        # Encode
        self.conv_encode1 = ML_ISTA_block(unfolding, 3, 64, 64)
        self.conv_pool1 = nn.MaxPool2d(2, 2)
        self.conv_encode2 = ML_ISTA_block(unfolding, 64, 128, 128)
        self.conv_pool2 = nn.MaxPool2d(2, 2)
        self.conv_encode3 = ML_ISTA_block(unfolding, 128, 256, 256)
        self.conv_pool3 = nn.MaxPool2d(2, 2)
        self.conv_encode4 = ML_ISTA_block(unfolding, 256, 512, 512)
        self.conv_pool4 = nn.MaxPool2d(2, 2)

        # Bottleneck
        self.bottleneck = ML_ISTA_block(unfolding, 512, 1024, 1024)

        # Decode
        self.conv_decode4 = expansive_block(unfolding, 1024, 512, 512)
        self.conv_decode3 = expansive_block(unfolding, 512, 256, 256)
        self.conv_decode2 = expansive_block(unfolding, 256, 128, 128)
        self.conv_decode1 = expansive_block(unfolding, 128, 64, 64)
        self.final_layer = final_block(64, out_channel)

    def forward(self, x):
        # Encode
        encode_block1 = self.conv_encode1(x)
        encode_pool1 = self.conv_pool1(encode_block1)
        encode_block2 = self.conv_encode2(encode_pool1)
        encode_pool2 = self.conv_pool2(encode_block2)
        encode_block3 = self.conv_encode3(encode_pool2)
        encode_pool3 = self.conv_pool3(encode_block3)
        encode_block4 = self.conv_encode4(encode_pool3)
        encode_pool4 = self.conv_pool4(encode_block4)

        # Bottleneck
        bottleneck = self.bottleneck(encode_pool4)

        # Decode
        decode_block4 = self.conv_decode4(encode_block4, bottleneck)
        decode_block3 = self.conv_decode3(encode_block3, decode_block4)
        decode_block2 = self.conv_decode2(encode_block2, decode_block3)
        decode_block1 = self.conv_decode1(encode_block1, decode_block2)
        final_layer = self.final_layer(decode_block1)

        return final_layer
