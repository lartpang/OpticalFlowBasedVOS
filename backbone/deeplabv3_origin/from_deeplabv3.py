# -*- coding: utf-8 -*-
# @Time    : 2020/8/5
# @Author  : Lart Pang
# @Email   : lartpang@163.com
# @File    : from_deeplabv3.py
# @Project : OpticalFlowBasedVOS
# @GitHub  : https://github.com/lartpang
import torch

from backbone.deeplabv3_origin.deeplabv3 import deeplabv3_resnet101

if __name__ == "__main__":
    in_C = 1
    in_data = torch.randn((4, 1, 320, 320))
    model = deeplabv3_resnet101(pretrained=True, progress=True,
                                in_channel=in_C, num_classes=None)
    inter_features = model(in_data)
    print([x.size() for x in inter_features.values()])
