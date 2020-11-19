# -*- coding: utf-8 -*-
# @Time    : 2020/9/18
# @Author  : Lart Pang
# @Email   : lartpang@163.com
# @File    : SharedModuleClass.py
# @Project : OpticalFlowBasedVOS
# @GitHub  : https://github.com/lartpang

import torch.nn as nn

from backbone.deeplabv3_origin.deeplabv3 import (
    deeplabv3_resnet101,
    deeplabv3_resnet50,
)
from backbone.origin.from_origin import (
    Backbone_R101_Custumed,
    Backbone_R50_Custumed,
)
from utils.misc import construct_print


class BasicResNetEncoder(nn.Module):
    def __init__(self, in_channel, model):
        super(BasicResNetEncoder, self).__init__()
        self.encoders = nn.ModuleList(model(in_channel))

    def forward(self, x):
        outs = []
        for en in self.encoders:
            x = en(x)
            outs.append(x)
        return outs


class BasicDeepLabV3Encoder(nn.Module):
    def __init__(self, in_channel, model, replace_stride_with_dilation):
        super(BasicDeepLabV3Encoder, self).__init__()
        assert isinstance(replace_stride_with_dilation, (tuple, list))
        assert len(replace_stride_with_dilation) == 3
        self.encoders = model(
            pretrained=True, progress=True, in_channel=in_channel,
            num_classes=None,
            replace_stride_with_dilation=replace_stride_with_dilation
        )

    def forward(self, x):
        outs = list(self.encoders(x).values())
        return outs


class BaseModel(nn.Module):
    def __init__(self, backbone_info: dict):
        super(BaseModel, self).__init__()
        self.pretrain_path = backbone_info.get('pretrain_path')

        if backbone_info['backbone'] == 'deeplabv3':
            depth = backbone_info['backbone_cfg'].get('depth', False)
            if not depth:
                depth = 101
            assert depth in [50, 101]
            if depth == 101:
                deeplab = deeplabv3_resnet101
            else:
                deeplab = deeplabv3_resnet50
            use_dilation = backbone_info['backbone_cfg'].get('use_dilation', (
                False, False, False))
            self.shared_encoder = BasicDeepLabV3Encoder(
                in_channel=3, model=deeplab,
                replace_stride_with_dilation=use_dilation
            )
        elif backbone_info['backbone'] == 'resnet':
            depth = backbone_info['backbone_cfg'].get('depth', False)
            if not depth:
                depth = 101
            assert depth in [50, 101]
            if depth == 101:
                resnet = Backbone_R101_Custumed
            else:
                resnet = Backbone_R50_Custumed
            self.shared_encoder = BasicResNetEncoder(in_channel=3, model=resnet)
        else:
            raise NotImplementedError

        if backbone_info['freeze_bn']:
            self.freeze_bn()

    def freeze_bn(self):
        construct_print("We will freeze all BN layers.")
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_grouped_params(self):
        """
        grouped_params = [
            {"params": params_groups['pretrained'], "lr": 0.1 * lr},
            {"params": params_groups['retrained'], "lr": lr},
        ]
        可以通过将参数划分为pretrained和retrained两个组，前者学习率会是后者的0.1倍
        """
        return None
