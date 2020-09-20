import torch
import torch.nn as nn
import torch.nn.functional as F

from model.BaseNet.net import BaseNet
from model.common import *

from .blocks import *


class ResNet(BaseNet):
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
