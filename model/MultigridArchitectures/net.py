import torch
import torch.nn as nn
import torch.nn.functional as F

from model.common import *

from .blocks import *
from .utils import *

from utils.config import Config


class MNANet(nn.Module):
    def __init__(
        self, name, block_type, layers, num_classes, residual=False, progressive=False,
    ):
        super().__init__()
        self.config = Config().get_instance()
        self.name = name
        self.writer = ""
        self.num_classes = num_classes
        self.channels_in = 64
        self.layer_set = layers
        self.residual = residual
        self.progressive = progressive
        self.block_type = block_type

        self.init_layer = init_block(self.channels_in)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.hidden_layers = self._build_layers(self.layer_set)

        self.global_maxpool = nn.AdaptiveMaxPool2d((1, 1))

        self.fc = nn.Linear(int(self.channels_in * 1.75), self.num_classes)

        self.inital_downsample_1 = interpolate(channels_in=64, scale=0.5)
        self.inital_downsample_2 = interpolate(channels_in=64, scale=0.25)

        self.scale = self.config["Misc"]["FC_Scale"]

        self._init_modules()

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
        hidden_layers = []
        for i in range(len(layers) - 1):
            hidden_layers.append(
                self._build_unit(
                    self.block_type, 64 * (2 ** i), layers[i], downsample=True, size=3,
                )
            )
            hidden_layers.append(transition())
        if self.block_type.expansion == 4:
            hidden_layers.append(
                self._build_unit(
                    self.block_type, self.channels_in, layers[-1], downsample=True, size=2,
                )
            )
        else:
            hidden_layers.append(
                self._build_unit(
                    self.block_type, self.channels_in, layers[-1], downsample=False, size=2,
                )
            )
        hidden_layers.append(transition())
        if self.progressive:
            hidden_layers[0] = mgconv_progressive_block(
                self.block_type, 64, 64, nlayers=layers[0],
            )
        return nn.Sequential(*hidden_layers)

    def _build_unit(
        self, block_type, channels, nblocks, downsample, size=3,
    ):

        layers = []
        layers.append(
            mgconv_base_block(
                block_type=block_type,
                channels_in=self.channels_in,
                channels_out=channels,
                size=size,
                residual=self.residual,
                downsample_identity=downsample,
            )
        )
        self.channels_in = channels * self.block_type.expansion
        for _ in range(1, nblocks):
            layers.append(
                mgconv_base_block(
                    block_type=block_type,
                    channels_in=self.channels_in,
                    channels_out=channels,
                    size=size,
                    residual=self.residual,
                )
            )
        return nn.Sequential(*layers)

    def _forward_impl(self, x):

        init = self.init_layer(x)

        out = [init, self.inital_downsample_1(init), self.inital_downsample_2(init)]

        out = self.hidden_layers(out)
        out = out[0]
        out = torch.flatten(out, 1)

        # out = self.global_maxpool(out)

        out = self.fc(out)

        out *= self.scale

        return out

    def forward(self, x):
        out = self._forward_impl(x)
        return out
