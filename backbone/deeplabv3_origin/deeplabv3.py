# -*- coding: utf-8 -*-
# @Time    : 2020/8/6
# @Author  : Lart Pang
# @Email   : lartpang@163.com
# @File    : deeplabv3.py
# @Project : OpticalFlowBasedVOS
# @GitHub  : https://github.com/lartpang
from collections import OrderedDict

import torch
from torch import nn
from torch.hub import load_state_dict_from_url
from torch.nn import functional as F
from torchvision.models._utils import IntermediateLayerGetter

from backbone.deeplabv3_origin import resnet
from backbone.utils import load_pretrained_params

model_urls = {
    "deeplabv3_resnet50_coco":
        "https://download.pytorch.org/models/deeplabv3_resnet50_coco"
        "-cd0a2569.pth",
    "deeplabv3_resnet101_coco":
        "https://download.pytorch.org/models/deeplabv3_resnet101_coco"
        "-586e9e4e.pth",
}


class DeepLabV3(nn.Module):
    """
    Implements DeepLabV3 model from
    `"Rethinking Atrous Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1706.05587>`_.
    """

    def __init__(self, backbone, classifier):
        super(DeepLabV3, self).__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x):
        # contract: features is a dict of tensors
        features = self.backbone(x)
        result = OrderedDict({k: v for k, v in features.items()})

        x = features["res4"]
        x = self.classifier(x)
        result["res4_aspp"] = x

        return result


class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        module_list = [ASPP(in_channels, [12, 24, 36])]
        if num_classes:
            module_list.extend([nn.Conv2d(256, 256, 3, padding=1, bias=False),
                                nn.BatchNorm2d(256),
                                nn.ReLU(),
                                nn.Conv2d(256, num_classes, 1)])
        super(DeepLabHead, self).__init__(*module_list)


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(
                in_channels, out_channels, 3, padding=dilation,
                dilation=dilation, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode="bilinear",
                             align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=256):
        super(ASPP, self).__init__()
        modules = []
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )
        )

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class FCNHead(nn.Sequential):
    def __init__(self, in_channels, channels):
        inter_channels = in_channels // 4
        layers = [
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1),
        ]

        super(FCNHead, self).__init__(*layers)


def _segm_resnet(backbone_name, in_channel, num_classes,
                 replace_stride_with_dilation):
    backbone = resnet.__dict__[backbone_name](
        in_channel=in_channel, pretrained=False,
        replace_stride_with_dilation=replace_stride_with_dilation
    )
    return_layers = {
        "relu": "conv",
        "layer1": "res1",
        "layer2": "res2",
        "layer3": "res3",
        "layer4": "res4",
    }
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    classifier = DeepLabHead(2048, num_classes)
    model = DeepLabV3(backbone, classifier)
    return model


def _load_model(backbone, pretrained, progress, in_channel, num_classes,
                replace_stride_with_dilation):
    model = _segm_resnet(backbone, in_channel, num_classes,
                         replace_stride_with_dilation)
    if pretrained:
        arch = "deeplabv3_" + backbone + "_coco"
        model_url = model_urls[arch]
        if model_url is None:
            raise NotImplementedError(
                "pretrained {} is not supported as of now".format(arch))
        else:
            print(f"Loading parameters from {model_url}")
            state_dict = load_state_dict_from_url(model_url, progress=progress)
            load_pretrained_params(model, state_dict)
    return model


def deeplabv3_resnet50(pretrained=False, progress=True, in_channel=3,
                       num_classes=None,
                       replace_stride_with_dilation=[False, False, False]):
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO
        train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to
        stderr
        num_classes (int) 如果仅想要直接输出aspp后的特征，则num_classes=None即可
    """
    return _load_model("resnet50", pretrained, progress, in_channel,
                       num_classes, replace_stride_with_dilation)


def deeplabv3_resnet101(pretrained=False, progress=True, in_channel=3,
                        num_classes=None,
                        replace_stride_with_dilation=(False, False, False)):
    """Constructs a DeepLabV3 model with a ResNet-101 backbone.

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO
        train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to
        stderr
        num_classes (int) 如果仅想要直接输出aspp后的特征，则num_classes=None即可
    """
    return _load_model("resnet101", pretrained, progress, in_channel,
                       num_classes, replace_stride_with_dilation)


if __name__ == "__main__":
    model = deeplabv3_resnet101(pretrained=True, progress=True)
    in_data = torch.rand((4, 3, 320, 320))
    print([v.size() for v in model(in_data).values()])
