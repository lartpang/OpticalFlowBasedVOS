# -*- coding: utf-8 -*-
# @Time    : 2020/10/22
# @Author  : Lart Pang
# @Email   : lartpang@163.com
# @File    : IOULoss.py
# @Project : CODProj
# @GitHub  : https://github.com/lartpang
import numpy as np
from scipy.ndimage import distance_transform_edt


def cal_iou_loss(seg_logits, seg_gts):
    assert seg_logits.shape == seg_gts.shape, (seg_logits.shape, seg_gts.shape)
    seg_probs = seg_logits.sigmoid()

    inter = (seg_probs * seg_gts).sum(dim=(1, 2, 3))
    union = (seg_probs + seg_gts).sum(dim=(1, 2, 3))
    iou = 1 - (inter + 1) / (union - inter + 1)
    return iou.mean()


def cal_weighted_iou_loss(seg_logits, seg_gts):
    """使用真值的 distance_transform_edt 来计算权重"""
    assert seg_logits.shape == seg_gts.shape, (seg_logits.shape, seg_gts.shape)
    seg_probs = seg_logits.sigmoid()

    total_weight = []
    gt_cpu = seg_gts.detach().cpu().squeeze().numpy()
    for g in gt_cpu:
        fore_region = distance_transform_edt(g)
        fore_region = (fore_region - fore_region.min()) / \
                      (fore_region.max() - fore_region.min())
        back_region = distance_transform_edt(1 - g)
        back_region = (back_region - back_region.min()) / \
                      (back_region.max() - back_region.min())
        edge_weight = 1 - (fore_region + back_region)
        total_weight.append(edge_weight)
    total_weight = np.array(total_weight)
    weight_tensor = seg_gts.new_tensor(total_weight).unsqueeze(1)

    inter = (weight_tensor * seg_probs * seg_gts).sum(dim=(1, 2, 3))
    union = (weight_tensor * (seg_probs + seg_gts)).sum(dim=(1, 2, 3))
    iou = 1 - (inter + 1) / (union - inter + 1)
    return iou.mean()
