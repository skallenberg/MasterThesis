import torch
import torch.nn as nn
import torch.nn.functional as F

from model.common import *
from model.ResNetv2.net import ResNetv2


class MRNNetv2(ResNetv2):
    def __init__(
        self, name, block_type, layers, num_classes, groups=1, width_per_group=64, depthwise=False
    ):
        super().__init__(
            name,
            block_type,
            layers,
            num_classes,
            groups=groups,
            width_per_group=width_per_group,
            depthwise=False,
        )

        self.downsample_0 = nn.Sequential(
            GhostBatchNorm(64, config["DataLoader"]["BatchSize"] // 32),
            nn.Conv2d(
                64,
                self.block_type.expansion * 64 * (2 ** (len(layers) - 1)),
                kernel_size=1,
                stride=1,
                bias=False,
            ),
            nn.MaxPool2d(kernel_size=3, stride=8, padding=1),
        )

    def _build_layers(self, layers):
        hidden_layers = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        hidden_layers.append(self._build_unit(64, layers[0]))
        for i in range(1, len(layers)):
            hidden_layers.append(self._build_unit(64 * (2 ** i), layers[i], stride=2,))
        channels_in = 64
        self.downsample_layers.append(
            nn.Sequential(
                GhostBatchNorm(channels_in, config["DataLoader"]["BatchSize"] // 32),
                nn.Conv2d(
                    channels_in,
                    64 * self.block_type.expansion,
                    kernel_size=1,
                    stride=1,
                    bias=False,
                ),
            )
        )
        channels_in = 64 * self.block_type.expansion
        for i in range(1, len(layers)):
            channels = 64 * (2 ** i)
            self.downsample_layers.append(
                nn.Sequential(
                    GhostBatchNorm(channels_in, config["DataLoader"]["BatchSize"] // 32),
                    nn.Conv2d(
                        channels_in,
                        channels * self.block_type.expansion,
                        kernel_size=1,
                        stride=1,
                        bias=False,
                    ),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                )
            )
            channels_in = channels * self.block_type.expansion

        return hidden_layers

    def _forward_impl(self, x):
        out = self.conv0(x)

        out = self.bn0(out)
        out = self.activation0(out)
        out = self.maxpool(out)

        identity_0 = out

        for idx, layer in enumerate(self.hidden_layers):
            identity_x = out
            out = layer(out)
            identity_x = self.downsample_layers[idx](identity_x)
            out = out + identity_x

        identity_0 = self.downsample_0(identity_0)
        out = out + identity_0
        out = self.global_maxpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        out *= self.scale
        return out
