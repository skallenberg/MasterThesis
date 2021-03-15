import torch
import torch.nn as nn
import torch.nn.functional as F

from model.BaseNet.net import BaseNet
from model.common import *

from .blocks import *

from utils.config import Config


class DenseNet(BaseNet):
    def __init__(
        self, name, block_type, layers, drop_rate=0, num_classes=10,
    ):
        self.growth_rate = 12
        self.drop_rate = drop_rate
        super().__init__(
            name, block_type, layers, num_classes,
        )

        self.fc = nn.Linear(self.nfeats, self.num_classes)

    def _build_layers(self, layers):

        nfeats = self.channels_in
        hidden_layers = []
        for i in range(len(layers)):
            hidden_layers.append(self._build_unit(self.block_type, layers[i], nfeats,))
            nfeats += layers[i] * self.growth_rate * self.block_type.expansion
            if i != len(layers) - 1:
                hidden_layers.append(self._transition(nfeats, int(nfeats / 2)))
                nfeats = int(nfeats / 2)

        if self.config["Misc"]["GhostBatchNorm"]:
            self.bnf = GhostBatchNorm(nfeats, self.config["DataLoader"]["BatchSize"] // 32)
        else:
            self.bnf = nn.BatchNorm2d(nfeats)

        if self.config["Misc"]["UseCELU"]:
            self.activationf = nn.CELU(self.config["Misc"]["CELU_alpha"])
        else:
            self.activationf = nn.ReLU(inplace=True)

        hidden_layers.append(self.bnf)
        hidden_layers.append(self.activationf)

        self.nfeats = nfeats

        self.hidden_layers = nn.Sequential(*hidden_layers)
        return nn.Sequential(*hidden_layers)

    def _build_unit(self, block_type, layer, channels):

        block = dense_block(
            block_type, layer, channels, growth_rate=self.growth_rate, drop_rate=self.drop_rate,
        )
        return block

    def _transition(self, channels_in, channels_out):
        if self.config["Misc"]["GhostBatchNorm"]:
            bn = GhostBatchNorm(channels_in, self.config["DataLoader"]["BatchSize"] // 32)
        else:
            bn = nn.BatchNorm2d(channels_in)

        if self.config["Misc"]["UseCELU"]:
            act = nn.CELU(self.config["Misc"]["CELU_alpha"])
        else:
            act = nn.ReLU(inplace=True)

        conv = conv_1x1(channels_in, channels_out)
        avg = nn.AvgPool2d(kernel_size=2, stride=2)

        return nn.Sequential(*[bn, act, conv, avg])
