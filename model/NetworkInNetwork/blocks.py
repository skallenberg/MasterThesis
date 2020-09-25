import torch
import torch.nn as nn
import torch.nn.functional as F

from model.BaseNet.blocks import *
from model.common import *

mlp_layers = 2


class NiN_base_block(base_block):
    expansion = 1

    def __init__(
        self, channels_in, channels_out, stride=1,
    ):
        super().__init__(
            channels_in, channels_out, stride=stride,
        )

        MLPconv = []
        for i in range(mlp_layers):
            MLPconv.append(conv_1x1(channels_out, channels_out))
        self.MLPconv = nn.Sequential(*MLPconv)

    def forward(self, x):

        out = self._forward_impl(x)
        out = self.MLPconv(out)

        return out


class NiN_bottleneck_block(bottleneck_block):
    expansion = 4

    def __init__(
        self, channels_in, channels_out, stride=1,
    ):
        super().__init__(
            channels_in, channels_out, stride=stride,
        )

        MLPconv = []
        for i in range(mlp_layers):
            MLPconv.append(conv_1x1(channels_out * self.expansion, channels_out * self.expansion,))
        self.MLPconv = nn.Sequential(*MLPconv)

    def forward(self, x):

        out = self._forward_impl(x)

        out = self.MLPconv(out)

        return out
