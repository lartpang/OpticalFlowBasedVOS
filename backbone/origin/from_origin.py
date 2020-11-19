import torch.nn as nn
import torchvision.models.resnet as R
import torchvision.models.vgg as V


def Backbone_V16_Custumed(in_C):
    net = V.vgg16_bn(pretrained=True, progress=True)
    if in_C == 3:
        div_1 = nn.Sequential(*list(net.children())[0][:6])
    else:
        div_1 = nn.Sequential(
            nn.Conv2d(in_C, 64, kernel_size=3, padding=1), *list(net.children())[0][1:6]
        )
    div_2 = nn.Sequential(*list(net.children())[0][6:13])
    div_4 = nn.Sequential(*list(net.children())[0][13:23])
    div_8 = nn.Sequential(*list(net.children())[0][23:33])
    div_16 = nn.Sequential(*list(net.children())[0][33:43])
    return div_1, div_2, div_4, div_8, div_16


def Backbone_R50_Custumed(in_C):
    net = R.resnet50(pretrained=True, progress=True)
    if in_C != 3:
        net.conv1 = nn.Conv2d(in_C, 64, kernel_size=7, stride=2, padding=3, bias=False)
    div_2 = nn.Sequential(*list(net.children())[:3])
    div_4 = nn.Sequential(*list(net.children())[3:5])
    div_8 = net.layer2
    div_16 = net.layer3
    div_32 = net.layer4

    return div_2, div_4, div_8, div_16, div_32


def Backbone_R101_Custumed(in_C):
    net = R.resnet101(pretrained=True, progress=True)
    if in_C != 3:
        net.conv1 = nn.Conv2d(in_C, 64, kernel_size=7, stride=2, padding=3, bias=False)
    div_2 = nn.Sequential(*list(net.children())[:3])
    div_4 = nn.Sequential(*list(net.children())[3:5])
    div_8 = net.layer2
    div_16 = net.layer3
    div_32 = net.layer4

    return div_2, div_4, div_8, div_16, div_32
