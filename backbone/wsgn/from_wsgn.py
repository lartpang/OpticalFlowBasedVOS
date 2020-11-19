from torch import nn

from backbone.wsgn.resnet import l_resnet101, l_resnet50
from backbone.wsgn import customized_func as L


def Backbone_WSGNR50_Custumed(in_C):
    net = l_resnet50(pretrained=True)
    if in_C != 3:
        net.conv1 = L.Conv2d(in_C, 64, kernel_size=7, stride=2, padding=3, bias=False)
    div_2 = nn.Sequential(*list(net.children())[:3])
    div_4 = nn.Sequential(*list(net.children())[3:5])
    div_8 = net.layer2
    div_16 = net.layer3
    div_32 = net.layer4

    return div_2, div_4, div_8, div_16, div_32


def Backbone_WSGNR101_Custumed(in_C):
    net = l_resnet101(pretrained=True)
    if in_C != 3:
        net.conv1 = L.Conv2d(in_C, 64, kernel_size=7, stride=2, padding=3, bias=False)
    div_2 = nn.Sequential(*list(net.children())[:3])
    div_4 = nn.Sequential(*list(net.children())[3:5])
    div_8 = net.layer2
    div_16 = net.layer3
    div_32 = net.layer4

    return div_2, div_4, div_8, div_16, div_32
