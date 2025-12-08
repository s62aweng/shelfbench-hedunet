import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import Convx2, DownBlock, UpBlock, WithSE


class MixtureUNet(nn.Module):
    """
    A straight-forward UNet implementation
    """

    def __init__(self, input_channels, mixture_modes=5, base_channels=16,
                 conv_block=Convx2, padding_mode='replicate', batch_norm=True,
                 squeeze_excitation=False):
        super().__init__()

        if squeeze_excitation:
            conv_block = WithSE(conv_block)

        bc = base_channels
        self.init = conv_block(input_channels, bc, bn=batch_norm, padding_mode=padding_mode)

        conv_args = dict(
            conv_block=conv_block,
            bn=batch_norm,
            padding_mode=padding_mode
        )

        self.down1 = DownBlock( 1 * bc,  2 * bc, **conv_args)
        self.down2 = DownBlock( 2 * bc,  4 * bc, **conv_args)
        self.down3 = DownBlock( 4 * bc,  8 * bc, **conv_args)
        self.down4 = DownBlock( 8 * bc, 16 * bc, **conv_args)
        self.down5 = DownBlock(16 * bc, 32 * bc, **conv_args)

        self.up1 = UpBlock(32 * bc, 16 * bc, **conv_args)
        self.up2 = UpBlock(16 * bc,  8 * bc, **conv_args)  # noqa: E201
        self.up3 = UpBlock( 8 * bc,  4 * bc, **conv_args)  # noqa: E201
        self.up4 = UpBlock( 4 * bc,  2 * bc, **conv_args)  # noqa: E201
        self.up5 = UpBlock( 2 * bc,  1 * bc, **conv_args)  # noqa: E201

        self.final = conv_block(bc, bc, bn=batch_norm, padding_mode=padding_mode)
        self.segment = nn.Conv2d(bc, mixture_modes, 1)

        self.mixture_coefficients = nn.Conv2d(32 * bc, mixture_modes, 4, dilation=4)

    def forward(self, x):
        x = self.init(x)

        skip1 = x
        x = self.down1(x)
        skip2 = x
        x = self.down2(x)
        skip3 = x
        x = self.down3(x)
        skip4 = x
        x = self.down4(x)
        skip5 = x
        x = self.down5(x)

        coeffs = self.mixture_coefficients(x)
        coeffs = F.avg_pool2d(coeffs, kernel_size=coeffs.shape[2:])
        coeffs = F.log_softmax(coeffs, dim=1)

        x = self.up1(x, skip5)
        x = self.up2(x, skip4)
        x = self.up3(x, skip3)
        x = self.up4(x, skip2)
        x = self.up5(x, skip1)

        x = self.final(x)
        x = self.segment(x)
        x = F.logsigmoid(x)

        prediction = torch.sum(torch.exp(coeffs + x), dim=1, keepdims=True)

        return prediction, coeffs, x
