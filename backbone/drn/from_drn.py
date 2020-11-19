import torch
from torch import nn

from backbone.drn.drn import drn_a_50


def Backbone_DRNA50_in3(in_C):
    net = drn_a_50(pretrained=True)
    if in_C != 3:
        net.conv1 = nn.Conv2d(in_C, 64, kernel_size=7, stride=2, padding=3, bias=False)
    div_2 = nn.Sequential(*list(net.children())[:3])
    div_4 = nn.Sequential(*list(net.children())[3:5])
    div_8 = net.layer2
    div_16 = net.layer3
    div_32 = net.layer4

    return div_2, div_4, div_8, div_16, div_32


if __name__ == "__main__":
    conv_list = Backbone_DRNA50_in3(in_C=3)
    x = torch.randn((1, 3, 320, 320))
    # torch.Size([1, 3, 320, 320])
    # torch.Size([1, 64, 160, 160])
    # torch.Size([1, 256, 80, 80])
    # torch.Size([1, 512, 40, 40])
    # torch.Size([1, 1024, 40, 40])
    for conv in conv_list:
        print(x.size())
        x = conv(x)
    print(x.size())
