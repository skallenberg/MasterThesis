import torch
import torch.nn as nn
import torch.nn.functional as F

from model.common import *
from .utils import *
from utils.config import Config

config = Config.get_instance()
alpha = 0.075

Λ = torch.load("./data/datasets/cifar10/eigens1.pt")
V = torch.load("./data/datasets/cifar10/eigens2.pt")


def whitening_block(c_in, c_out, eps=1e-2):
    filt = nn.Conv2d(3, 27, kernel_size=(3, 3), padding=(1, 1), bias=False)
    filt.weight.data = V / torch.sqrt(Λ + eps)[:, None, None, None]
    filt.weight.requires_grad = False

    return nn.Sequential(
        filt,
        nn.Conv2d(27, c_out, kernel_size=(1, 1), bias=False),
        GhostBatchNorm(c_out, 2),
        nn.CELU(alpha),
    )


class base_block(nn.Module):
    expansion = 1

    def __init__(
        self, channels_in, channels_out, stride=1, groups=1, base_width=64, dilation=1, extra=False
    ):
        super().__init__()
        self.stride = stride
        self.channels_out = channels_out
        self.bn = GhostBatchNorm(channels_out, config["DataLoader"]["BatchSize"] // 32)
        self.conv = conv_3x3(channels_in, channels_out)
        self.activation = nn.CELU(alpha)
        if stride > 1:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            self.pool = lambda x: x

        if extra:
            self.extra = nn.Sequential(
                nn.Conv2d(channels_out, channels_out, kernel_size=3),
                GhostBatchNorm(channels_out, config["DataLoader"]["BatchSize"] // 32),
                self.activation,
            )
        else:
            self.extra = lambda x: x
        if groups != 1 or base_width != 64:
            raise ValueError("base_block only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in base_block")

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

    def __init__(self, channels_in, channels_out, stride=1, groups=1, base_width=64, dilation=1):
        super().__init__()
        width = int(channels_out * (base_width / 64.0)) * groups
        self.bn1 = nn.BatchNorm2d(width)
        self.conv1 = conv_1x1(channels_in, width)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv2 = conv_3x3(width, width, stride=stride, groups=groups, dilation=dilation)
        self.bn3 = nn.BatchNorm2d(channels_out * self.expansion)
        self.conv3 = conv_1x1(width, channels_out * self.expansion)
        self.activation = nn.CELU(alpha)

    def _forward_impl(self, x):
        out = self.conv1(out)
        out = self.bn1(x)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.activation(out)
        return out

    def forward(self, x):
        out = self._forward_impl(x)
        return out
