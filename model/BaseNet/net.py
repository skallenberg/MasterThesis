import torch
import torch.nn as nn
import torch.nn.functional as F

from model.common import *

from .blocks import *


class BaseNet(nn.Module):
    def __init__(
        self,
        name,
        block_type,
        layers,
        num_classes,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        rgb=False,
    ):
        super().__init__()

        if isinstance(layers, int):
            layers = [1] * layers
        if replace_stride_with_dilation is None:
            self.replace_stride_with_dilation = [False] * (len(layers))

        if len(self.replace_stride_with_dilation) != len(layers):
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a n-element tuple, got {}".format(self.replace_stride_with_dilation)
            )

        self.name = name
        self.writer = ""
        self.num_classes = num_classes
        self.dilation = 1
        self.block_type = block_type
        self.groups = groups
        self.base_width = width_per_group
        self.block_type = block_type
        self.channels_in = 64
        self.conv0 = initial_conv(self.channels_in)
        self.bn0 = nn.BatchNorm2d(self.channels_in)
        self.activation0 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer_set = layers

        self.hidden_layers = self._build_layers(self.layer_set)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.block_type.expansion * 64 * (2 ** (len(layers) - 1)), num_classes)

        self._init_modules()

    def _init_modules(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _build_layers(self, layers):
        hidden_layers = [self._build_unit(64, layers[0])]
        for i in range(1, len(layers)):
            hidden_layers.append(
                self._build_unit(
                    64 * (2 ** i),
                    layers[i],
                    stride=2,
                    dilate=self.replace_stride_with_dilation[i],
                )
            )
        return nn.Sequential(*hidden_layers)

    def _build_unit(self, channels, blocks, stride=1, dilate=False):
        previous_dilation = self.dilation

        if dilate:
            self.dilation *= stride
            stride = 1

        layers = []
        layers.append(
            self.block_type(
                channels_in=self.channels_in,
                channels_out=channels,
                stride=stride,
                groups=self.groups,
                base_width=self.base_width,
                dilation=previous_dilation,
            )
        )
        self.channels_in = channels * self.block_type.expansion
        for _ in range(1, blocks):
            layers.append(
                self.block_type(
                    channels_in=self.channels_in,
                    channels_out=channels,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=previous_dilation,
                )
            )
        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        out = self.conv0(x)
        out = self.bn0(out)
        out = self.activation0(out)
        out = self.maxpool(out)

        out = self.hidden_layers(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out

    def forward(self, x):
        out = self._forward_impl(x)
        return out
