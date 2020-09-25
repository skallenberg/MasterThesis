from numpy.core.numeric import identity
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.common import *
from model.ResNet.net import ResNet


class MRNNet(ResNet):
    def __init__(self, name, block_type, layers, num_classes):
        super().__init__(
            name, block_type, layers, num_classes,
        )
        if config["Misc"]["GhostBatchNorm"]:
            bn0 = GhostBatchNorm(64, config["DataLoader"]["BatchSize"] // 32)
        else:
            bn0 = nn.BatchNorm2d(64)
        self.downsample_0 = nn.Sequential(
            bn0,
            conv_1x1(64, self.block_type.expansion * 64 * (2 ** (len(layers) - 1))),
            nn.MaxPool2d(kernel_size=3, stride=8, padding=1),
        )

    def _build_layers(self, layers):
        hidden_layers = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()

        hidden_layers.append(self._build_unit(64, layers[0]))
        for i in range(1, len(layers)):
            hidden_layers.append(self._build_unit(64 * (2 ** i), layers[i], stride=2,))

        channels_in = 64

        if config["Misc"]["GhostBatchNorm"]:
            bn = GhostBatchNorm(channels_in, config["DataLoader"]["BatchSize"] // 32)
        else:
            bn = nn.BatchNorm2d(channels_in)
        self.downsample_layers.append(
            nn.Sequential(bn, conv_1x1(channels_in, 64 * self.block_type.expansion,),)
        )

        channels_in = 64 * self.block_type.expansion
        for i in range(1, len(layers)):
            channels = 64 * (2 ** i)
            if config["Misc"]["GhostBatchNorm"]:
                bn = GhostBatchNorm(channels_in, config["DataLoader"]["BatchSize"] // 32)
            else:
                bn = nn.BatchNorm2d(channels_in)
            self.downsample_layers.append(
                nn.Sequential(
                    bn,
                    conv_1x1(channels_in, channels * self.block_type.expansion,),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                )
            )
            channels_in = channels * self.block_type.expansion

        return hidden_layers

    def _forward_impl(self, x):
        out = self.init_layer(x)
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
