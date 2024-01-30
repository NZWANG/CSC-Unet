# -*- encoding: utf-8 -*-
# Author  : haitong
# Time    : 2024-01-27 16:15
# File    : CSCUNetPP.py
# Software: PyCharm


import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

__all__ = ['UNetPP']


class CSCBlock(nn.Module):

    def __init__(self, in_channels, mid_channels, out_channels, t=1):
        super(CSCBlock, self).__init__()
        # print(f"\n\tunfoldings:{T}")

        self.T = t
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


class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class UNetPP(nn.Module):
    def __init__(self, num_classes, input_channels=3):
        super().__init__()
        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0] * 2 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1] * 2 + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2] * 2 + nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0] * 3 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1] * 3 + nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0] * 4 + nb_filter[1], nb_filter[0], nb_filter[0])

        # deep_supervision
        self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input):  # [1 3 640 480]
        x0_0 = self.conv0_0(input)  
        x1_0 = self.conv1_0(self.pool(x0_0))  # [1 64 320 240]
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))  

        x2_0 = self.conv2_0(self.pool(x1_0))  # [1 128 160 120]
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))  # [1 64 320 240]
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))  # [1 32 640 480]

        x3_0 = self.conv3_0(self.pool(x2_0))  # [1 256 80 60]
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))  # [1 128 160 120]
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))  # [1 64 320 240]
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))  

        x4_0 = self.conv4_0(self.pool(x3_0))  # [1 512 40 30]
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))  
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))  
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))  # [1 64 320 240]
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1)) 

        output4 = self.final4(x0_4)  

        if self.training:
            output1 = self.final1(x0_1)  
            output3 = self.final3(x0_3)  
            return [output1, output2, output3, output4]

        else:
            return output4



class CSCUNetPP(nn.Module):
    def __init__(self, num_classes, unfolding, input_channels=3):
        super().__init__()
        nb_filter = [32, 64, 128, 256, 512]
        self.unfolding = unfolding
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = CSCBlock(input_channels, nb_filter[0], nb_filter[0], unfolding)
        self.conv1_0 = CSCBlock(nb_filter[0], nb_filter[1], nb_filter[1], unfolding)
        self.conv2_0 = CSCBlock(nb_filter[1], nb_filter[2], nb_filter[2], unfolding)
        self.conv3_0 = CSCBlock(nb_filter[2], nb_filter[3], nb_filter[3], unfolding)
        self.conv4_0 = CSCBlock(nb_filter[3], nb_filter[4], nb_filter[4], unfolding)

        self.conv0_1 = CSCBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0], unfolding)
        self.conv1_1 = CSCBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1], unfolding)
        self.conv2_1 = CSCBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2], unfolding)
        self.conv3_1 = CSCBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3], unfolding)

        self.conv0_2 = CSCBlock(nb_filter[0] * 2 + nb_filter[1], nb_filter[0], nb_filter[0], unfolding)
        self.conv1_2 = CSCBlock(nb_filter[1] * 2 + nb_filter[2], nb_filter[1], nb_filter[1], unfolding)
        self.conv2_2 = CSCBlock(nb_filter[2] * 2 + nb_filter[3], nb_filter[2], nb_filter[2], unfolding)

        self.conv0_3 = CSCBlock(nb_filter[0] * 3 + nb_filter[1], nb_filter[0], nb_filter[0], unfolding)
        self.conv1_3 = CSCBlock(nb_filter[1] * 3 + nb_filter[2], nb_filter[1], nb_filter[1], unfolding)

        self.conv0_4 = CSCBlock(nb_filter[0] * 4 + nb_filter[1], nb_filter[0], nb_filter[0], unfolding)

        # deep_supervision
        self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input):  
        x0_0 = self.conv0_0(input)  
        x1_0 = self.conv1_0(self.pool(x0_0))  # [1 64 320 240]
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))  

        x2_0 = self.conv2_0(self.pool(x1_0)) 
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))  # [1 64 320 240]
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1)) 

        x3_0 = self.conv3_0(self.pool(x2_0))  
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))  # [1 128 160 120]
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))  
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))  

        x4_0 = self.conv4_0(self.pool(x3_0))  
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1)) 
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))  
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))  
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))  

        output4 = self.final4(x0_4)  

        if self.training:
            output1 = self.final1(x0_1)  
            output2 = self.final2(x0_2)  
            output3 = self.final3(x0_3)  
            return [output1, output2, output3, output4]

        else:
            return output4







