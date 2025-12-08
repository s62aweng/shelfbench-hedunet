"""Follows https://github.com/shsun-xq/DeepUNet/blob/master/trainSealand.py"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNRelu(nn.Module):
    def __init__(self, c_in, c_out, relu=True):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, 3, padding=1)
        # self.bn = nn.BatchNorm2d(c_out)
        self.relu = relu

    def forward(self, x):
        # x = self.bn(self.conv(x))
        x = self.conv(x)
        if self.relu:
            x = F.relu(x)
        return x


class DownBlock(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.conv1 = ConvBNRelu(f, 2*f)
        self.conv2 = ConvBNRelu(2*f, f, relu=False)

    def forward(self, x):
        """this doesn't match the network diagrams,
        but it is a direct port of the linked github..."""
        x = F.max_pool2d(x, 2)
        temp = self.conv1(x)
        bn = self.conv2(temp) + x
        act = F.relu(bn)

        return bn, act


class UpBlock(nn.Module):
    """Follows Fig. 6 in the paper"""
    def __init__(self, f):
        super().__init__()
        self.conv1 = ConvBNRelu(2*f, 2*f)
        self.conv2 = ConvBNRelu(2*f, f, relu=False)

    def forward(self, act, bn):
        x = F.interpolate(act, scale_factor=2)
        temp = self.conv1(torch.cat([bn, x], dim=1))
        bn = self.conv2(temp) + x
        act = F.relu(bn)

        return act


class DeepUNet(nn.Module):
    """
    A straight-forward UNet implementation
    """
    tasks = ['seg']

    def __init__(self):
        super().__init__()
        self.init1 = ConvBNRelu(2, 32)
        self.init2 = ConvBNRelu(32, 64)
        self.init3 = ConvBNRelu(64, 32, relu=False)

        self.down1 = DownBlock(32)
        self.down2 = DownBlock(32)
        self.down3 = DownBlock(32)
        self.down4 = DownBlock(32)
        self.down5 = DownBlock(32)
        self.down6 = DownBlock(32)

        self.up1 = UpBlock(32)
        self.up2 = UpBlock(32)
        self.up3 = UpBlock(32)
        self.up4 = UpBlock(32)
        self.up5 = UpBlock(32)
        self.up6 = UpBlock(32)

        self.classify = nn.Conv2d(32, 1, 1)

    def forward(self, x, side):
        x = self.init1(x)
        x = self.init2(x)
        bn1 = self.init3(x)
        act1 = F.relu(bn1)

        bn2, act2 = self.down1(act1)
        bn3, act3 = self.down1(act2)
        bn4, act4 = self.down1(act3)
        bn5, act5 = self.down1(act4)
        bn6, act6 = self.down1(act5)
        bn7, act7 = self.down1(act6)

        temp = self.up1(act7, bn6)
        temp = self.up2(temp, bn5)
        temp = self.up3(temp, bn4)
        temp = self.up4(temp, bn3)
        temp = self.up5(temp, bn2)
        temp = self.up6(temp, bn1)

        out = self.classify(temp)
        return out
