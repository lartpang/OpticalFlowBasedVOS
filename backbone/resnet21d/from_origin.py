# -*- coding: utf-8 -*-
# @Time    : 2020/9/28
# @Author  : Lart Pang
# @Email   : lartpang@163.com
# @File    : from_origin.py
# @Project : OpticalFlowBasedVOS
# @GitHub  : https://github.com/lartpang
import torch
from torchvision.models.video import r2plus1d_18


def Backbone_R2Plus1D18_Custumed(in_C):
    assert in_C == 3
    model = r2plus1d_18(pretrained=True, progress=True)
    div_2 = model.stem
    div_4 = model.layer2
    div_8 = model.layer3
    div_16 = model.layer4

    return div_2, div_4, div_8, div_16


if __name__ == '__main__':
    data = torch.randn(4, 3, 10, 320, 320)
    for m in Backbone_R2Plus1D18_Custumed(3):
        print(data.shape)
        data = m(data)
    print(data.shape)
