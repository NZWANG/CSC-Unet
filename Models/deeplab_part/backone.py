# -*- encoding: utf-8 -*-
# Author  : haitong
# Time    : 2024-01-24 15:53
# File    : backone.py
# Software: PyCharm


from Models.deeplab_part import resnet, xception, cscvgg

def build_backbone(backbone, output_stride, BatchNorm, unfolding):
    if backbone == 'resnet101':
        return resnet.ResNet101(output_stride, BatchNorm)
    elif backbone == 'xception':
        return xception.AlignedXception(output_stride, BatchNorm)
    elif backbone == 'cscvgg':
        return cscvgg.CSCVGG(unfolding)
    elif backbone == 'vgg':
        return cscvgg.VGG()

    else:
        raise NotImplementedError
