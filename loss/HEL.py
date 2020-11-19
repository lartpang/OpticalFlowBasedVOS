# -*- coding: utf-8 -*-
# @Time    : 2019/8/24 下午10:02
# @Author  : Lart Pang
# @FileName: EdgeRegionLoss.py
# @Project : USVideoSeg
# @GitHub  : https://github.com/lartpang
import torch.nn.functional as F


def cal_hel_loss(seg_logits, seg_gts):
    """
    from the HEL in Hierarchical Dynamic Filtering Network for RGB-D Salient Object Detection, ECCV 2020.
    https://github.com/lartpang/HDFNet/blob/master/loss/HEL.py
    """

    def _edge_loss(pred, target):
        edge = target - F.avg_pool2d(target, kernel_size=5, stride=1, padding=2)
        edge[edge != 0] = 1
        numerator = (edge * (pred - target).abs()).sum([2, 3])
        denominator = edge.sum([2, 3]) + 1e-6
        return numerator / denominator

    def _region_loss(pred, target):
        # 该部分损失更强调前景区域内部或者背景区域内部的预测一致性
        numerator_fore = (target - target * pred).sum([2, 3])
        denominator_fore = target.sum([2, 3]) + 1e-6

        numerator_back = ((1 - target) * pred).sum([2, 3])
        denominator_back = (1 - target).sum([2, 3]) + 1e-6
        return numerator_fore / denominator_fore + numerator_back / denominator_back

    seg_logits = seg_logits.sigmoid()
    edge_loss = _edge_loss(seg_logits, seg_gts)
    region_loss = _region_loss(seg_logits, seg_gts)
    return (edge_loss + region_loss).mean()
