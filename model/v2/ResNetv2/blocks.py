import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.v2.BaseNetv2.blocks import *
from model.common import *


class residual_base_block(base_block):
    expansion = 1

    def __init__(
        self,
        channels_in,
        channels_out,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dropout=0.5,
        depthwise=False,
    ):
        super().__init__(
            channels_in,
            channels_out,
            stride=stride,
            groups=groups,
            base_width=base_width,
            depthwise=False,
        )
        self.downsample = downsample
        self.dropout = dropout

    def forward(self, x):
        skip = np.random.binomial(n=1, p=self.dropout)
        identity = x

        if skip == 0:
            if self.downsample is not None:
                identity = self.downsample(x)

            out = identity
        else:
            out = self._forward_impl(x)

            if self.downsample is not None:
                identity = self.downsample(x)

            out = out + identity
        return out


class residual_bottleneck_block(bottleneck_block):
    expansion = 4

    def __init__(
        self,
        channels_in,
        channels_out,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dropout=0.5,
        depthwise=False,
    ):
        super().__init__(
            channels_in,
            channels_out,
            stride=stride,
            groups=groups,
            base_width=base_width,
            depthwise=False,
        )
        self.downsample = downsample
        self.dropout = dropout

    def forward(self, x):
        skip = np.random.binomial(n=1, p=self.dropout)
        identity = x

        if skip == 0:
            if self.downsample is not None:
                identity = self.downsample(x)

            out = identity
        else:
            out = self._forward_impl(x)

            if self.downsample is not None:
                identity = self.downsample(x)

            out = out + identity
        return out
