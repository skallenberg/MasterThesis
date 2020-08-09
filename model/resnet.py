import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.basenet import base_net, _base_block, _bottleneck_block


class _residual_base_block(_base_block):
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
        dropout=0.5,
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


class _residual_bottleneck_block(_bottleneck_block):
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
        dropout=0.5,
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


class ResNet(base_net):
    def __init__(
        self,
        name,
        block_type,
        layers,
        num_classes,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        sd=None,
    ):

        self.nlayers = sum(layers)
        self.layer_count = 0

        if sd == "const":
            self.layer_prob = self._sd_prob(mode="const")
        elif sd == "prog":
            self.layer_prob = self._sd_prob(mode="prog")
        else:
            self.layer_prob = self._sd_prob(mode="none")

        super().__init__(
            name,
            block_type,
            layers,
            num_classes,
            groups=groups,
            width_per_group=width_per_group,
            replace_stride_with_dilation=replace_stride_with_dilation,
        )

    def _sd_prob(self, mode="const", least_prob=0.5):
        if mode == "const":
            prob = lambda x: 0.5
        elif mode == "prog":
            prob = lambda x: 1 - x / self.nlayers * (1 - least_prob)
        else:
            prob = lambda x: 1
        return prob

    def _build_unit(self, channels, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation

        if dilate:
            self.dilation *= stride
            stride = 1

        if stride != 1 or self.channels_in != channels * self.block_type.expansion:
            downsample = nn.Sequential(
                nn.BatchNorm2d(self.channels_in),
                nn.Conv2d(
                    self.channels_in,
                    channels * self.block_type.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
            )

        layers = [
            (
                self.block_type(
                    channels_in=self.channels_in,
                    channels_out=channels,
                    stride=stride,
                    downsample=downsample,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=previous_dilation,
                    dropout=self.layer_prob(self.layer_count),
                )
            )
        ]
        self.layer_count += 1
        self.channels_in = channels * self.block_type.expansion
        for _ in range(1, blocks):
            layers.append(
                self.block_type(
                    channels_in=self.channels_in,
                    channels_out=channels,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=previous_dilation,
                    dropout=self.layer_prob(self.layer_count),
                )
            )
            self.layer_count += 1
        return nn.Sequential(*layers)


def _resnet(name, block_type, layers, num_classes=10, **kwargs):
    model = ResNet(name, block_type, layers, num_classes, **kwargs)
    return model


def ResBaseTest():
    return _resnet("ResBaseTest", _residual_base_block, [1, 1])


def ResBottleTest():
    return _resnet("ResBottleTest", _residual_bottleneck_block, [1, 1])


def StochasticDepthTest():
    return _resnet("ResBaseTest", _residual_base_block, [1, 1], sd="prog")


def ResNet34():
    return _resnet("ResNet34", _residual_base_block, [3, 4, 6, 3])


def ResNet50():
    return _resnet("ResNet50", _residual_bottleneck_block, [3, 4, 6, 3])


def NeXtTest(**kwargs):
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 4
    return _resnet("NeXtTest", _residual_bottleneck_block, [1, 1], **kwargs)


def ResNeXt50(**kwargs):
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 4
    return _resnet("ResNeXt50", _residual_bottleneck_block, [3, 4, 6, 3], **kwargs)


def WideTest(**kwargs):
    kwargs["width_per_group"] = 64 * 2
    return _resnet("WideTest", _residual_bottleneck_block, [1, 1], **kwargs)


def WideResNet50(**kwargs):
    kwargs["width_per_group"] = 64 * 2
    return _resnet("WideResNet50", _residual_bottleneck_block, [3, 4, 6, 3], **kwargs)

