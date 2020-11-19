# -*- coding: utf-8 -*-
# @Time    : 2020/11/15
# @Author  : Lart Pang
# @Email   : lartpang@163.com
# @File    : utils.py.py
# @Project : OpticalFlowBasedVOS
# @GitHub  : https://github.com/lartpang
import torch.nn as nn


def load_pretrained_params(model: nn.Module, pretrained_dict: dict):
    print("Loading the pretrianed parameters...")
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    if list(pretrained_dict.keys())[0].startswith("module."):
        pretrained_dict = {
            k[7:]: v
            for k, v in pretrained_dict.items()
            if (k[7:] in model_dict) and (v.size() == model_dict[k[7:]].size())
        }
    else:
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if (k in model_dict) and (v.size() == model_dict[k].size())
        }
    print(
        "the number of public keys: ",
        len(pretrained_dict.keys()),
        "\nthe number of private keys of model: ",
        len(set(pretrained_dict.keys()).difference(set(model_dict.keys()))),
    )
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    print("Loaded the pretrianed parameters...")
