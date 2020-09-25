import torch
import torch.nn as nn
import torch.nn.functional as F

from model.common import *


class base_block(nn.Module):
    expansion = 1

    def __init__(
        self, channels_in, channels_out, stride=1,
    ):
        super().__init__()
        self.layer = conv_bn_act(channels_in, channels_out, stride=stride)

    def _forward_impl(self, x):
        out = self.layer(x)
        return out

    def forward(self, x):
        out = self._forward_impl(x)
        return out


class bottleneck_block(nn.Module):
    expansion = 4

    def __init__(
        self, channels_in, channels_out, stride=1,
    ):
        super().__init__()
        self.layer = conv_bn_act_bottleneck(channels_in, channels_out, stride)

    def _forward_impl(self, x):
        out = self.layer(x)
        return out

    def forward(self, x):
        out = self._forward_impl(x)
        return out
