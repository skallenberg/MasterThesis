import torch
import torch.nn as nn
import torch.nn.functional as F
from model.common import *
from model.BaseNet.blocks import *

global mlp_layers
mlp_layers = 2


class NiN_base_block(base_block):
    expansion = 1

    def __init__(
        self,
        channels_in,
        channels_out,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
    ):
        super().__init__(
            channels_in,
            channels_out,
            stride=stride,
            groups=groups,
            base_width=base_width,
            dilation=dilation,
        )
        self.downsample = downsample

        MLPconv = []
        for i in range(mlp_layers):
            MLPconv.append(nn.Conv2d(channels_out, channels_out, kernel_size=1))
        self.MLPconv = nn.Sequential(*MLPconv)

    def forward(self, x):

        out = self._forward_impl(x)
        out = self.MLPconv(out)

        return out


class NiN_bottleneck_block(bottleneck_block):
    expansion = 4

    def __init__(
        self,
        channels_in,
        channels_out,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
    ):
        super().__init__(
            channels_in,
            channels_out,
            stride=stride,
            groups=groups,
            base_width=base_width,
            dilation=dilation,
        )

        MLPconv = []
        for i in range(mlp_layers):
            MLPconv.append(
                nn.Conv2d(
                    channels_out * self.expansion,
                    channels_out * self.expansion,
                    kernel_size=1,
                )
            )
        self.MLPconv = nn.Sequential(*MLPconv)

    def forward(self, x):

        out = self._forward_impl(x)

        out = self.MLPconv(out)

        return out
