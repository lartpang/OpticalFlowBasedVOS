# -*- coding: utf-8 -*-
# @Time    : 2020
# @Author  : Lart Pang
# @FileName: BaseOps.py
# @GitHub  : https://github.com/lartpang

import torch
import torch.nn.functional as F
from torch import nn


def cus_sample(feat: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    :param feat: 输入特征
    :param kwargs: size或者scale_factor
    """
    assert len(kwargs.keys()) == 1 and list(kwargs.keys())[0] in [
        "size",
        "scale_factor",
    ]
    if size := kwargs.get('size', False):
        if tuple(size) == feat.shape[2:]:
            return feat
    if scale_factor := kwargs.get('scale_factor', False):
        if scale_factor == 1:
            return feat
        if isinstance(scale_factor, float):
            kwargs['recompute_scale_factor'] = False
    return F.interpolate(feat, **kwargs, mode="bilinear", align_corners=False)


def upsample_add(*xs: torch.Tensor) -> torch.Tensor:
    y = xs[-1]
    for x in xs[:-1]:
        y = y + F.interpolate(x, size=y.size()[2:], mode="bilinear",
                              align_corners=False)
    return y


def upsample_cat(*xs):
    y = xs[-1]
    out = []
    for x in xs[:-1]:
        out.append(F.interpolate(x, size=y.size()[2:], mode="bilinear",
                                 align_corners=False))
    return torch.cat([*out, y], dim=1)


def upsample_reduce(b, a):
    """
    上采样所有特征到最后一个特征的尺度以及前一个特征的通道数
    """
    _, C, _, _ = b.size()
    N, _, H, W = a.size()

    b = F.interpolate(b, size=(H, W), mode="bilinear", align_corners=False)
    a = a.reshape(N, -1, C, H, W).mean(1)

    return b + a


def shuffle_channels(x, groups):
    """
    Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,W] -> [N,C,H,W]
    一共C个channel要分成g组混合的channel，先把C reshape成(g, C/g)的形状，
    然后转置成(C/g, g)最后平坦成C组channel
    """
    N, C, H, W = x.size()
    x = x.reshape(N, groups, C // groups, H, W).permute(0, 2, 1, 3, 4)
    return x.reshape(N, C, H, W)


def is_on_gpu(x):
    """
    判定x是否是gpu上的实例，可以检测tensor和module
    :param x: (torch.Tensor, nn.Module)目标对象
    :return: 是否在gpu上
    """
    # https://blog.csdn.net/WYXHAHAHA123/article/details/86596981
    if isinstance(x, torch.Tensor):
        return "cuda" in x.device
    elif isinstance(x, nn.Module):
        return next(x.parameters()).is_cuda
    else:
        raise NotImplementedError


def get_device(x):
    """
    返回x的设备信息，可以处理tensor和module
    :param x: (torch.Tensor, nn.Module) 目标对象
    :return: 所在设备
    """
    # https://blog.csdn.net/WYXHAHAHA123/article/details/86596981
    if isinstance(x, torch.Tensor):
        return x.device
    elif isinstance(x, nn.Module):
        return next(x.parameters()).device
    else:
        raise NotImplementedError


def slide_inference(model, ori_data, win_size, overlap_ratio, model_kwargs):
    """Inference by sliding-window with overlap.
    :param ori_data: dict(image=..., flow=...)
    :param win_size: int/tuple/list the size of the crop fed into the model

    Example:
        outputs = slide_inference(
            model=model,
            ori_data=dict(curr_jpeg=curr_jpegs, curr_flow=curr_flows),
            win_size=(320, 320),
            overlap_ratio=1 / 2,
            model_kwargs=dict(return_sigmoid=False)
        )
    """
    if isinstance(win_size, int):
        win_size = (win_size, win_size)
    assert isinstance(win_size, (tuple, list))
    h_crop, w_crop = win_size

    N, _, H, W = ori_data["curr_jpeg"].size()
    h_stride = int(h_crop * (1 - overlap_ratio))
    w_stride = int(w_crop * (1 - overlap_ratio))
    assert h_crop <= H and w_crop <= W, (
        'crop size should not greater than image size')

    h_grids = max(H - h_crop + h_stride - 1, 0) // h_stride + 1
    w_grids = max(W - w_crop + w_stride - 1, 0) // w_stride + 1

    full_logits = ori_data["curr_jpeg"].new_zeros((N, 1, H, W))
    count_mat = ori_data["curr_jpeg"].new_zeros((N, 1, H, W))
    for h_idx in range(h_grids):
        for w_idx in range(w_grids):
            y1 = h_idx * h_stride
            x1 = w_idx * w_stride
            y2 = min(y1 + h_crop, H)
            x2 = min(x1 + w_crop, W)
            y1 = max(y2 - h_crop, 0)
            x1 = max(x2 - w_crop, 0)
            crop_jpegs = ori_data["curr_jpeg"][:, :, y1:y2, x1:x2]
            crop_flows = ori_data["curr_flow"][:, :, y1:y2, x1:x2]
            model_output = model(curr_jpeg=crop_jpegs, curr_flow=crop_flows,
                                 **model_kwargs)
            crop_logits = model_output['curr_seg']
            full_logits += F.pad(
                crop_logits, [int(x1), int(W - x2), int(y1), int(H - y2)],
                mode='constant', value=0
            )
            count_mat[:, :, y1:y2, x1:x2] += 1

    assert (count_mat == 0).sum() == 0
    full_logits = full_logits / count_mat  # 通过count_mat来对重叠区域取均值
    return full_logits


class InputPadder:
    def get_pad_list(self, curr_shape, target_shape):
        curr_h, curr_w = curr_shape[-2:]
        target_h, target_w = target_shape[-2:]
        assert target_h > curr_h and target_w > curr_w, (target_h, curr_h,
                                                         target_w, curr_w)
        if target_h != curr_h or target_w != curr_w:
            pad_ht = target_h - curr_h
            pad_wd = target_w - curr_w
            self._pad = [
                pad_wd // 2,
                pad_wd - pad_wd // 2,
                pad_ht // 2,
                pad_ht - pad_ht // 2,
            ]

    def pad(self, x, target_shape):
        self.get_pad_list(x.shape, target_shape)
        return F.pad(x, self._pad, mode="replicate")

    def unpad_tensors(self, xs):
        assert isinstance(xs, (list, tuple))
        return [self.unpad_tensor(x) for x in xs]

    def unpad_tensor(self, x):
        assert isinstance(x, torch.Tensor)
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3],
             self._pad[0], wd - self._pad[1]]
        return x[..., c[0]: c[1], c[2]: c[3]]


if __name__ == "__main__":
    a = torch.rand(3, 4, 10, 10)
    b = torch.rand(3, 2, 5, 5)
    print(upsample_reduce(b, a).size())
