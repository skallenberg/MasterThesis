import torch
import torch.nn as nn
import torch.nn.functional as F

from model.common import *

from .blocks import *
from utils.config import Config


class BaseNet(nn.Module):
    def __init__(
        self, name, block_type, layers, num_classes,
    ):
        super().__init__()

        self.config = Config().get_instance()

        self.name = name
        self.writer = ""
        self.num_classes = num_classes
        self.block_type = block_type
        self.channels_in = 64

        self.init_layer = init_block(self.channels_in)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer_set = layers

        self.hidden_layers = self._build_layers(self.layer_set)

        self.global_maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Linear(self.block_type.expansion * 64 * (2 ** (len(layers) - 1)), num_classes)

        self.scale = self.config["Misc"]["FC_Scale"]

        #self._init_modules()

    def _init_modules(self):
        if self.config["Misc"]["UseCELU"]:
            nonlinearity = "leaky_relu"
        else:
            nonlinearity = "relu"
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity=nonlinearity)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if self.config["Misc"]["GhostBatchNorm"]:
                    pass
                else:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def _build_layers(self, layers):
        hidden_layers = [self._build_unit(64, layers[0], stride=1)]
        for i in range(1, len(layers)):
            hidden_layers.append(self._build_unit(64 * (2 ** i), layers[i], stride=2))
        return nn.Sequential(*hidden_layers)

    def _build_unit(
        self, channels, blocks, stride=1,
    ):
        layers = []
        layers.append(
            self.block_type(channels_in=self.channels_in, channels_out=channels, stride=stride)
        )
        self.channels_in = channels * self.block_type.expansion
        for _ in range(1, blocks):
            layers.append(self.block_type(channels_in=self.channels_in, channels_out=channels))
        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        out = self.init_layer(x)
        out = self.maxpool(out)

        out = self.hidden_layers(out)

        out = self.global_maxpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        out *= self.scale

        return out

    def forward(self, x):
        out = self._forward_impl(x)
        return out
