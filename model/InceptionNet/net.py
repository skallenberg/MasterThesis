import torch
import torch.functional as F
import torch.nn as nn

from .blocks import *


class interpolate(nn.Module):
    def __init__(self, channels_in=None, scale=None, mode="nearest"):
        super().__init__()
        self.interpol = nn.functional.interpolate
        self.scale = scale
        self.mode = mode
        self.channels_in = channels_in
        if self.channels_in:
            self.conv = conv_1x1(
                channels_in=self.channels_in, channels_out=int(self.channels_in * self.scale),
            )

    def forward(self, x):
        out = self.interpol(x, scale_factor=self.scale, mode=self.mode)
        if self.channels_in:
            out = self.conv(out)

        return out


class Inception_v4(nn.Module):
    def __init__(self, name, layers, num_classes=10):
        super().__init__()

        self.name = name
        self.writer = ""
        self.init_interpolate = interpolate(scale=9.34375)

        hidden_layers = []
        hidden_layers.append(stem_v2())
        for i in range(layers[0]):
            hidden_layers.append(InceptionA(channels_in=384))

        hidden_layers.append(ReductionA(channels_in=384, k=192, l=224, m=256, n=384))

        for i in range(layers[1]):
            hidden_layers.append(InceptionB(channels_in=1024))

        hidden_layers.append(ReductionB(channels_in=1024))

        for i in range(layers[2]):
            hidden_layers.append(InceptionC(channels_in=1536))

        self.hidden_layers = nn.Sequential(*hidden_layers)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.dropout = nn.Dropout2d(0.8)

        self.fc = nn.Linear(1536, num_classes)

    def _forward_impl(self, x):
        out = x

        out = self.init_interpolate(out)

        out = self.hidden_layers(out)

        out = self.avgpool(out)

        out = self.dropout(out)

        out = torch.flatten(out, 1)

        out = self.fc(out)

        return out

    def forward(self, x):
        out = self._forward_impl(x)
        return out
