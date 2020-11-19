# -*- coding: utf-8 -*-
# @Time    : 2020/11/15
# @Author  : Lart Pang
# @Email   : lartpang@163.com
# @File    : BasicModules.py
# @Project : OpticalFlowBasedVOS
# @GitHub  : https://github.com/lartpang
from torch import nn

from module.BasicBlocks import BasicConv2d
from utils.tensor_ops import cus_sample, upsample_add


class BasicEncoder(nn.Module):
    def __init__(self, backbone):
        super(BasicEncoder, self).__init__()
        self.encoders = nn.ModuleList(backbone(3))

    def forward(self, x):
        outs = []
        for en in self.encoders:
            x = en(x)
            outs.append(x)
        return outs


class BasicTranslayer(nn.Module):
    def __init__(self, out_c):
        super(BasicTranslayer, self).__init__()
        self.c5_down = nn.Conv2d(2048, out_c, 1)
        self.c4_down = nn.Conv2d(1024, out_c, 1)
        self.c3_down = nn.Conv2d(512, out_c, 1)
        self.c2_down = nn.Conv2d(256, out_c, 1)
        self.c1_down = nn.Conv2d(64, out_c, 1)

    def forward(self, xs):
        assert isinstance(xs, (tuple, list))
        assert len(xs) == 5
        c1, c2, c3, c4, c5 = xs
        c5 = self.c5_down(c5)
        c4 = self.c4_down(c4)
        c3 = self.c3_down(c3)
        c2 = self.c2_down(c2)
        c1 = self.c1_down(c1)
        outs = [c5, c4, c3, c2, c1]
        return outs


class BasicSegHead(nn.Module):
    def __init__(self, mid_c):
        super(BasicSegHead, self).__init__()
        self.p5_d5 = BasicConv2d(mid_c, mid_c, 3, 1, 1)
        self.p4_d4 = BasicConv2d(mid_c, mid_c, 3, 1, 1)
        self.p3_d3 = BasicConv2d(mid_c, mid_c, 3, 1, 1)
        self.p2_d2 = BasicConv2d(mid_c, mid_c, 3, 1, 1)
        self.p1_d1 = BasicConv2d(mid_c, mid_c, 3, 1, 1)
        self.seg_d0 = nn.Sequential(
            BasicConv2d(mid_c, 32, 3, 1, 1),
            nn.Conv2d(32, 1, 1)
        )

    def forward(self, xs):
        p5, p4, p3, p2, p1 = xs

        d5 = self.p5_d5(p5)
        d4 = self.p4_d4(upsample_add(d5, p4))
        d3 = self.p3_d3(upsample_add(d4, p3))
        d2 = self.p2_d2(upsample_add(d3, p2))
        d1 = self.p1_d1(upsample_add(d2, p1))
        d0 = cus_sample(d1, scale_factor=2)

        s_0 = self.seg_d0(d0)
        return s_0
