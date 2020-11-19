import os
from functools import partial

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW, lr_scheduler, SGD

from utils.misc import construct_print


def save_checkpoint(exp_name, model, current_epoch, full_net_path, state_net_path, optimizer, scheduler, scaler=None):
    """
    保存完整参数模型（大）和状态参数模型（小）

    Args:
        current_epoch (int): 当前周期
        full_net_path (str): 保存完整参数模型的路径
        state_net_path (str): 保存模型权重参数的路径
    """
    state_dict = dict(
        arch=exp_name,
        epoch=current_epoch,
        net_state=model.state_dict(),
        opti_state=optimizer.state_dict(),
        sche_state=scheduler.state_dict() if scaler else None,
        scaler=scaler.state_dict(),
    )
    torch.save(state_dict, full_net_path)
    torch.save(model.state_dict(), state_net_path)


def resume_checkpoint(exp_name, load_path, model, optimizer=None, scheduler=None, scaler=None, mode="all",
                      force_load=False):
    """
    从保存节点恢复模型

    Args:
        load_path (str): 模型存放路径
        model: your model
        optimizer: your optimizer
        mode (str): 选择哪种模型恢复模式:
            - 'all': 回复完整模型，包括训练中的的参数, will return start_epoch；
            - 'onlynet': 仅恢复模型权重参数
    """
    assert os.path.exists(load_path) and os.path.isfile(load_path), load_path

    construct_print(f"Loading checkpoint '{load_path}'")
    checkpoint = torch.load(load_path)

    if mode == "all":
        assert (optimizer is not None) and (scheduler is not None)
        if exp_name == checkpoint["arch"] or force_load:
            start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["net_state"])
            optimizer.load_state_dict(checkpoint["opti_state"])
            scheduler.load_state_dict(checkpoint["sche_state"])
            if scaler and checkpoint.get('scaler', None) is not None:
                scaler.load_state_dict(checkpoint["scaler"])
            construct_print(
                f"Loaded '{load_path}' " f"(epoch {checkpoint['epoch']})")
        else:
            raise Exception(f"{load_path} does not match.")
        return start_epoch
    elif mode == "onlynet":
        model.load_state_dict(checkpoint)
        construct_print(
            f"Loaded checkpoint '{load_path}' " f"(only has the net's weight "
            f"params)")
    else:
        raise NotImplementedError


def _get_lr_coefficient(curr_epoch, total_num, lr_strategy, scheduler_cfg):
    # because the parameter `total_num` is involved in assignment,
    # so, if we don't want to pass this varible through the function
    # _get_lr_coefficient's parameters, we need to use the nonlocal keyword.

    # ** curr_epoch start from 0 **
    if lr_strategy == "poly":
        turning_epoch = scheduler_cfg["warmup_length"]
        if curr_epoch < turning_epoch:
            # 0,1,2,...,turning_epoch-1
            coefficient = 1 / turning_epoch * (1 + curr_epoch)
        else:
            # turning_epoch,...,end_epoch
            curr_epoch -= turning_epoch - 1
            total_num -= turning_epoch - 1
            coefficient = np.power((1 - float(curr_epoch) / total_num),
                                   scheduler_cfg["lr_decay"])
        if min_coef := scheduler_cfg.get('min_coef'):
            coefficient = max(min_coef, coefficient)
    elif lr_strategy == "cos":
        # \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 +
        # \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right)
        # c_t = c_{min} + 1/2 * (c_max - c_min) * (1 + cos(i_cur / i_max * pi))
        turning_epoch = scheduler_cfg["warmup_length"]
        if curr_epoch < turning_epoch:
            # 0,1,2,...,turning_epoch-1
            coefficient = 1 / turning_epoch * (1 + curr_epoch)
        else:
            # turning_epoch,...,end_epoch
            curr_epoch -= turning_epoch - 1
            total_num -= turning_epoch - 1
            min_coef = scheduler_cfg['min_coef']
            max_coef = scheduler_cfg['max_coef']
            coefficient = min_coef + (max_coef - min_coef) * \
                          (1 + np.cos(np.pi * curr_epoch / total_num)) / 2
    elif lr_strategy == "linearonclr":
        coefficient = 1 - np.abs((curr_epoch + 1) / (total_num + 1) * 2 - 1)
    else:
        raise Exception(f"{lr_strategy} is not implemented")

    return coefficient


def make_optimizer_with_cfg(model: nn.Module, optimizer_cfg: dict):
    lr = optimizer_cfg['lr']
    optimizer_strategy = optimizer_cfg['strategy']
    optimizer_type = optimizer_cfg["optimizer"]
    chosen_optimizer_cfg = optimizer_cfg[optimizer_type]

    if optimizer_strategy == "trick":
        # https://github.com/implus/PytorchInsight/blob/master
        # /classification/imagenet_tricks.py
        grouped_params = [
            {
                "params": [
                    p for name, p in model.named_parameters()
                    if ("bias" in name or "bn" in name)
                ],
                "weight_decay": 0,
            },
            {
                "params": [p for name, p in model.named_parameters()
                           if ("bias" not in name and "bn" not in name)]
            },
        ]
    elif optimizer_strategy == "r3":
        grouped_params = [
            # 不对bias参数执行weight decay操作，weight decay主要的作用就是通过对网络
            # 层的参数（包括weight和bias）做约束（L2正则化会使得网络层的参数更加平滑）达
            # 到减少模型过拟合的效果。
            {
                "params": [
                    param for name, param in model.named_parameters()
                    if name[-4:] == "bias"
                ],
                "lr": 2 * lr,
            },
            {
                "params": [
                    param for name, param in model.named_parameters()
                    if name[-4:] != "bias"
                ],
                "lr": lr,
                "weight_decay": chosen_optimizer_cfg['weight_decay'],
            },
        ]
    elif optimizer_strategy == "all":
        grouped_params = model.parameters()
    elif optimizer_strategy == "finetune":
        params_groups = model.get_grouped_params()
        if params_groups is not None:
            grouped_params = [
                {"params": params_groups['pretrained'], "lr": 0.1 * lr},
                {"params": params_groups['retrained'], "lr": lr},
            ]
        else:
            grouped_params = [{"params": model.parameters(), 'lr': 0.1 * lr}]
    elif optimizer_strategy == "f3":
        backbone, head = [], []
        for name, params_tensor in model.named_parameters():
            if name.startswith("shared_encoder.encoders.0"):
                pass
            elif name.startswith("shared_encoder"):
                backbone.append(params_tensor)
            else:
                head.append(params_tensor)
        grouped_params = [
            {"params": backbone, "lr": 0.1 * lr},
            {"params": head, "lr": lr},
        ]
    else:
        raise NotImplementedError

    if optimizer_type == 'sgd':
        optimizer = SGD(params=grouped_params,
                        lr=lr,
                        momentum=chosen_optimizer_cfg['momentum'],
                        weight_decay=chosen_optimizer_cfg['weight_decay'],
                        nesterov=chosen_optimizer_cfg['nesterov'])
    elif optimizer_type == 'adamw':
        optimizer = AdamW(params=grouped_params,
                          lr=lr,
                          weight_decay=chosen_optimizer_cfg['weight_decay'],
                          eps=1e-8)
    else:
        raise NotImplementedError
    return optimizer


def make_scheduler_with_cfg(optimizer, total_num, scheduler_cfg: dict):
    lr_strategy = scheduler_cfg["lr_strategy"]
    chosen_scheduler_cfg = scheduler_cfg[lr_strategy]
    if lr_strategy == "clr":
        # # cycle_id表示当前处于第几个cycle中，这里的cycle_id从1开始计数
        # # 这里的step_size表示半个cycle对应的迭代次数
        # cycle_id = np.floor(1 + curr_epoch / (2 * step_size))
        # # 这里实际上在判定当前处于cycle中的位置所对应的lr尺度，是一个 ^ 形状的折线
        # x = 1 - np.abs(curr_epoch / step_size - 2 * cycle_id + 1)
        # lr = base_lr + (max_lr - base_lr) * np.maximum(0, x)
        scheduler = lr_scheduler.CyclicLR(
            optimizer=optimizer,
            base_lr=chosen_scheduler_cfg["min_lr"],
            max_lr=chosen_scheduler_cfg["max_lr"],
            step_size_up=chosen_scheduler_cfg["step_size"],
            scale_mode=chosen_scheduler_cfg["mode"],
        )
    elif lr_strategy == 'step':
        scheduler = lr_scheduler.MultiStepLR(
            optimizer=optimizer,
            milestones=chosen_scheduler_cfg['milestones'],
            gamma=chosen_scheduler_cfg['gamma']
        )
    else:
        lr_func = partial(_get_lr_coefficient,
                          total_num=total_num,
                          lr_strategy=lr_strategy,
                          scheduler_cfg=chosen_scheduler_cfg)
        scheduler = lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lr_func)
    return scheduler


def _normalize(data_array: np.ndarray) -> np.ndarray:
    max_pred_array = data_array.max()
    min_pred_array = data_array.min()
    if max_pred_array != min_pred_array:
        data_array = (data_array - min_pred_array) / \
                     (max_pred_array - min_pred_array)
    return data_array


def clip_to_normalize(data_array: np.ndarray, clip_range: tuple = None) -> np.ndarray:
    clip_range = sorted(clip_range)
    if len(clip_range) == 3:
        clip_min, clip_mid, clip_max = clip_range
        assert 0 <= clip_min < clip_mid < clip_max <= 1, clip_range
        lower_array = data_array[data_array < clip_mid]
        higher_array = data_array[data_array > clip_mid]
        if lower_array.size > 0:
            lower_array = np.clip(lower_array, a_min=clip_min, a_max=1)
            max_lower = lower_array.max()
            lower_array = _normalize(lower_array) * max_lower
            data_array[data_array < clip_mid] = lower_array
        if higher_array.size > 0:
            higher_array = np.clip(higher_array, a_min=0, a_max=clip_max)
            min_lower = higher_array.min()
            higher_array = _normalize(higher_array) * (
                    1 - min_lower) + min_lower
            data_array[data_array > clip_mid] = higher_array
    elif len(clip_range) == 2:
        clip_min, clip_max = clip_range
        assert 0 <= clip_min < clip_max <= 1, clip_range
        if clip_min != 0 and clip_max != 1:
            data_array = np.clip(data_array, a_min=clip_min, a_max=clip_max)
        data_array = _normalize(data_array)
    elif clip_range is None:
        data_array = _normalize(data_array)
    else:
        raise NotImplementedError
    return data_array


def clip_normalize_scale(array, clip_min=0, clip_max=250, new_min=0, new_max=255):
    array = np.clip(array, a_min=clip_min, a_max=clip_max)
    array = (array - array.min()) / (array.max() - array.min())
    array = array * (new_max - new_min) + new_min
    return array
