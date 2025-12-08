import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import Convx2, DownBlock, UpBlock, WithSE


class PDenseNet(nn.Module):
    def __init__(self, input_channels, dense_channels=8):
        super().__init__()

        c = dense_channels
        n_dense_channels = [
            input_channels,
            input_channels + c,
            input_channels + 2*c,
            input_channels + 2*c,
            input_channels + 4*c,
            input_channels + 5*c,
        ]

        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.BatchNorm2d(c_in),
                nn.PReLU(c_in),
                nn.Conv2d(c_in, c, 3, padding=1)
            ) for c_in in n_dense_channels])

        self.final = nn.Sequential(
            nn.Conv2d(input_channels+6*c, input_channels, 1),
            nn.BatchNorm2d(input_channels)
        )

    def forward(self, x):
        x0 = x

        x1 = self.blocks[0](x0)
        x2 = self.blocks[1](torch.cat([x0, x1], dim=1))
        x3 = self.blocks[2](torch.cat([x0, x1, x2], dim=1))
        x4 = self.blocks[3](torch.cat([x0, x2, x3], dim=1))
        x5 = self.blocks[4](torch.cat([x0, x1, x2, x3, x4], dim=1))
        x6 = self.blocks[5](torch.cat([x0, x1, x2, x3, x4, x4], dim=1))
        x_dense = self.final(torch.cat([x0, x1, x2, x3, x4, x5, x6], dim=1))

        return x0 + x_dense


def BN_PReLU_Conv(channels):
    return nn.Sequential(
        nn.BatchNorm2d(channels), 
        nn.PReLU(channels),
        nn.Conv2d(channels, channels, 3, padding=1)
    )


class DeepStructureUNet(nn.Module):
    """
    https://arxiv.org/pdf/2003.07784.pdf
    """
    tasks = ['seg']

    def __init__(self, input_channels, output_channels=1, base_channels=64):
        super().__init__()
        bc = base_channels

        self.down1 = nn.Sequential(
            nn.Conv2d(input_channels, bc, 3, padding=1),
            PDenseNet(bc),
            BN_PReLU_Conv(bc)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(bc, 2*bc, 2, stride=2),
            BN_PReLU_Conv(2*bc),
            PDenseNet(2*bc),
            BN_PReLU_Conv(2*bc)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(2*bc, 4*bc, 2, stride=2),
            BN_PReLU_Conv(4*bc),
            PDenseNet(4*bc),
            BN_PReLU_Conv(4*bc),
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(4*bc, 8*bc, 2, stride=2),
            BN_PReLU_Conv(8*bc),
            PDenseNet(8*bc),
            BN_PReLU_Conv(8*bc),
        )

        self.bridge = nn.Sequential(
            nn.Conv2d(8*bc, 16*bc, 2, stride=2),
            BN_PReLU_Conv(16*bc),
            PDenseNet(16*bc),
            BN_PReLU_Conv(16*bc),
        )

        self.unpool4 = nn.ConvTranspose2d(16*bc, 8*bc, 2, stride=2)
        self.up4 = nn.Sequential(
            BN_PReLU_Conv(8*bc),
            PDenseNet(8*bc),
            BN_PReLU_Conv(8*bc),
        )

        self.unpool3 = nn.ConvTranspose2d(8*bc, 4*bc, 2, stride=2)
        self.up3 = nn.Sequential(
            BN_PReLU_Conv(4*bc),
            PDenseNet(4*bc),
            BN_PReLU_Conv(4*bc),
        )

        self.unpool2 = nn.ConvTranspose2d(4*bc, 2*bc, 2, stride=2)
        self.up2 = nn.Sequential(
            BN_PReLU_Conv(2*bc),
            PDenseNet(2*bc),
            BN_PReLU_Conv(2*bc),
        )

        self.unpool1 = nn.ConvTranspose2d(2*bc, 1*bc, 2, stride=2)
        self.up1 = nn.Sequential(
            BN_PReLU_Conv(1*bc),
            PDenseNet(1*bc),
            BN_PReLU_Conv(1*bc),
        )

        self.final = nn.Conv2d(bc, output_channels, 1)

    def forward(self, x, *args):
        x = x1 = self.down1(x)
        x = x2 = self.down2(x)
        x = x3 = self.down3(x)
        x = x4 = self.down4(x)
        x = self.bridge(x)
        x = self.up4(self.unpool4(x) + x4)
        x = self.up3(self.unpool3(x) + x3)
        x = self.up2(self.unpool2(x) + x2)
        x = self.up1(self.unpool1(x) + x1)

        return self.final(x)
