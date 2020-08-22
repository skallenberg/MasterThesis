import torch
import torch.nn as nn
import torch.nn.functional as F

from model.common import *


class base_block(nn.Module):
    expansion = 1

    def __init__(
        self, channels_in, channels_out, stride=1, groups=1, base_width=64, dilation=1,
    ):
        super().__init__()
        self.stride = 1
        self.bn1 = nn.BatchNorm2d(channels_in)
        self.conv1 = conv_3x3(channels_in, channels_out, stride)
        self.bn2 = nn.BatchNorm2d(channels_out)
        self.conv2 = conv_3x3(channels_out, channels_out)
        self.activation1 = nn.ReLU(inplace=True)
        self.activation2 = nn.ReLU(inplace=True)

        if groups != 1 or base_width != 64:
            raise ValueError("base_block only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in base_block")

    def _forward_impl(self, x):
        out = self.bn1(x)
        out = self.activation1(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.activation2(out)
        out = self.conv2(out)
        return out

    def forward(self, x):
        out = self._forward_impl(x)
        return out


class bottleneck_block(nn.Module):
    expansion = 4

    def __init__(self, channels_in, channels_out, stride=1, groups=1, base_width=64, dilation=1):
        super().__init__()
        width = int(channels_out * (base_width / 64.0)) * groups
        self.bn1 = nn.BatchNorm2d(channels_in)
        self.conv1 = conv_1x1(channels_in, width)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv2 = conv_3x3(width, width, stride=stride, groups=groups, dilation=dilation)
        self.bn3 = nn.BatchNorm2d(width)
        self.conv3 = conv_1x1(width, channels_out * self.expansion)
        self.activation1 = nn.ReLU(inplace=True)
        self.activation2 = nn.ReLU(inplace=True)
        self.activation3 = nn.ReLU(inplace=True)

    def _forward_impl(self, x):
        out = self.bn1(x)
        out = self.activation1(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.activation2(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.activation3(out)
        out = self.conv3(out)
        return out

    def forward(self, x):
        out = self._forward_impl(x)
        return out
