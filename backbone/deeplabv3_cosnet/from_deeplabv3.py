# -*- coding: utf-8 -*-
# @Time    : 2020/8/10
# @Author  : Lart Pang
# @Email   : lartpang@163.com
# @File    : from_deeplabv3.py.py
# @Project : OpticalFlowBasedVOS
# @GitHub  : https://github.com/lartpang
import torch
from torch import nn

from backbone.deeplabv3_cosnet.deeplabv3 import Res_Deeplab


def Backbone_DLV3_Custumed(in_C, pretrained=True):
    net = Res_Deeplab(pretrained=pretrained)
    if in_C != 3:
        net.conv1 = nn.Conv2d(in_C, 64, kernel_size=7, stride=2, padding=3, bias=False)
    div_2 = nn.Sequential(*list(net.children())[:3])
    div_4 = nn.Sequential(*list(net.children())[3:5])
    div_8 = net.layer2
    div_16 = net.layer3
    div_32 = net.layer4
    div_32_aspp = net.layer5

    return div_2, div_4, div_8, div_16, div_32, div_32_aspp


if __name__ == "__main__":
    x = torch.randn((3, 3, 320, 320))
    for m in Backbone_DLV3_Custumed(in_C=3):
        print(x.size())
        x = m(x)
    print(x.size())
