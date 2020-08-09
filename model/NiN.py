import torch
import torch.nn as nn
import torch.nn.functional as F
from model.basenet import base_net, _base_block, _bottleneck_block

global mlp_layers
mlp_layers = 2


class _NiN_base_block(_base_block):
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


class _NiN_bottleneck_block(_bottleneck_block):
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
            downsample=None,
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


class NiN_Net(base_net):
    def __init__(
        self,
        name,
        block_type,
        layers,
        num_classes,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        full_connect=False,
    ):
        super().__init__(
            name,
            block_type,
            layers,
            num_classes,
            groups=groups,
            width_per_group=width_per_group,
            replace_stride_with_dilation=replace_stride_with_dilation,
        )
        self.full_connect = full_connect

    def _forward_impl(self, x):
        out = self.conv0(x)
        out = self.bn0(out)
        out = self.activation0(out)
        out = self.maxpool(out)

        out = self.hidden_layers(out)

        if not self.full_connect:
            out = out.mean([2, 3])
        else:
            out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out


def _NiN_Net(name, block_type, layers, num_classes=10):
    model = NiN_Net(name, block_type, layers, num_classes)
    return model


def NiN_BaseTest():
    return _NiN_Net("NiN_BaseTest", _NiN_base_block, [1, 1])


def NiN_BottleTest():
    return _NiN_Net("NiN_BottleTest", _NiN_bottleneck_block, [1, 1])


def NiN_Net34():
    return _NiN_Net("NiN_Net34", _NiN_base_block, [3, 4, 6, 3])


def NiN_Net50():
    return _NiN_Net("NiN_Net50", _NiN_bottleneck_block, [3, 4, 6, 3])
