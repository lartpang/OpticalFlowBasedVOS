# -*- coding: utf-8 -*-
# @Time    : 2020
# @Author  : Lart Pang
# @FileName: BaseBlocks.py
# @GitHub  : https://github.com/lartpang

import torch.nn as nn


class BasicConv2d(nn.Module):
    def __init__(
            self,
            in_planes,
            out_planes,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
    ):
        super(BasicConv2d, self).__init__()

        self.basicconv = nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            ),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.basicconv(x)


class CustomizedBasicConv2d(nn.Module):
    def __init__(
            self,
            in_planes,
            out_planes,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
            has_bn=True,
            has_relu=True,
    ):
        super(CustomizedBasicConv2d, self).__init__()

        self.basicconv = nn.Sequential()
        self.basicconv.add_module(
            name="conv",
            module=nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            ),
        )
        if has_bn:
            self.basicconv.add_module(name="bn", module=nn.BatchNorm2d(out_planes)),
        if has_relu:
            self.basicconv.add_module(name="relu", module=nn.ReLU(inplace=True))

    def forward(self, x):
        return self.basicconv(x)
