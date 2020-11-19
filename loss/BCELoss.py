# -*- coding: utf-8 -*-
# @Time    : 2020/10/22
# @Author  : Lart Pang
# @Email   : lartpang@163.com
# @File    : BCE.py
# @Project : CODProj
# @GitHub  : https://github.com/lartpang
import torch.nn.functional as F


def cal_bce_loss(seg_logits, seg_gts):
    assert seg_logits.shape == seg_gts.shape, (seg_logits.shape, seg_gts.shape)
    return F.binary_cross_entropy_with_logits(
        input=seg_logits, target=seg_gts, reduction='mean'
    )
