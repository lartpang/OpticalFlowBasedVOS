# -*- coding: utf-8 -*-
# @Time    : 2020/10/22
# @Author  : Lart Pang
# @Email   : lartpang@163.com
# @File    : L12Loss.py
# @Project : CODProj
# @GitHub  : https://github.com/lartpang

def cal_mse_loss(seg_logits, seg_gts):
    assert seg_logits.shape == seg_gts.shape, (seg_logits.shape, seg_gts.shape)
    seg_probs = seg_logits.sigmoid()

    loss_map = (seg_probs - seg_gts).pow(2)
    return loss_map.mean()


def cal_mae_loss(seg_logits, seg_gts):
    assert seg_logits.shape == seg_gts.shape, (seg_logits.shape, seg_gts.shape)
    assert seg_logits.dtype == seg_gts.dtype
    seg_probs = seg_logits.sigmoid()

    loss_map = (seg_probs - seg_gts).abs()
    return loss_map.mean()
