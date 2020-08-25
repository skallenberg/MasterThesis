import torch
import torch.nn as nn
import torch.nn.functional as F

from model.common import *
from utils.config import Config

config = Config.get_instance()
alpha = config["Misc"]["CELU_alpha"]


class base_block(nn.Module):
    expansion = 1

    def __init__(
        self,
        channels_in,
        channels_out,
        stride=1,
        groups=1,
        base_width=64,
        depthwise=False,
        extra=False,
    ):
        super().__init__()
        width = int(channels_out * (base_width / 64.0)) * groups
        if depthwise:
            width = channels_out
            groups = channels_in
        self.extra_val = extra
        self.stride = stride
        self.conv = conv_3x3(channels_in, width, groups=groups)
        self.bn = GhostBatchNorm(width, config["DataLoader"]["BatchSize"] // 32)
        self.activation = nn.CELU(alpha)
        if stride > 1:
            self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.pool = lambda x: x

        if self.extra_val:
            if depthwise:
                groups = channels_out
            self.extra = nn.Sequential(
                conv_3x3(width, channels_out, groups=groups),
                GhostBatchNorm(channels_out, config["DataLoader"]["BatchSize"] // 32),
                self.activation,
            )
        else:
            self.extra = lambda x: x

    def _forward_impl(self, x):
        out = self.conv(x)
        out = self.pool(out)
        out = self.bn(out)
        out = self.activation(out)
        out = self.extra(out)
        return out

    def forward(self, x):
        out = self._forward_impl(x)
        return out


class bottleneck_block(nn.Module):
    expansion = 4

    def __init__(self, channels_in, channels_out, stride=1, groups=1, base_width=64, extra=False):
        super().__init__()
        width = int(channels_out * (base_width / 64.0)) * groups
        self.conv1 = conv_1x1(channels_in, width, groups=groups)
        self.bn1 = GhostBatchNorm(width, config["DataLoader"]["BatchSize"] // 32)

        self.conv2 = conv_3x3(width, width, groups=groups)
        self.bn2 = GhostBatchNorm(width, config["DataLoader"]["BatchSize"] // 32)

        self.conv3 = conv_1x1(width, channels_out * self.expansion, groups=groups)
        self.bn3 = GhostBatchNorm(
            channels_out * self.expansion, config["DataLoader"]["BatchSize"] // 32
        )
        self.activation = nn.CELU(alpha)

        if stride > 1:
            self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.pool = lambda x: x

    def _forward_impl(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.pool(out)
        out = self.bn2(out)
        out = self.activation(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.activation(out)
        return out

    def forward(self, x):
        out = self._forward_impl(x)
        return out
