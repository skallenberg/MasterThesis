import torch
import torch.nn as nn
import torch.nn.functional as F

from model.v2.BaseNetv2.net import BaseNetv2
from model.common import *

from .blocks import *

from utils.config import Config

config = Config.get_instance()
alpha = config["Misc"]["CELU_alpha"]


class DenseNetv2(BaseNetv2):
    def __init__(
        self,
        name,
        block_type,
        layers,
        drop_rate=0,
        num_classes=10,
        groups=1,
        width_per_group=64,
        depthwise=False,
    ):
        self.growth_rate = 32
        self.drop_rate = drop_rate
        super().__init__(
            name,
            block_type,
            layers,
            num_classes,
            groups=groups,
            width_per_group=width_per_group,
            depthwise=depthwise,
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

        self.bnf = GhostBatchNorm(nfeats, config["DataLoader"]["BatchSize"] // 32)
        self.activationf = nn.CELU(alpha)

        hidden_layers.append(self.bnf)
        hidden_layers.append(self.activationf)

        self.nfeats = nfeats

        self.hidden_layers = nn.Sequential(*hidden_layers)
        return nn.Sequential(*hidden_layers)

    def _build_unit(self, block_type, layer, channels):

        block = dense_block(
            block_type,
            layer,
            channels,
            growth_rate=self.growth_rate,
            drop_rate=self.drop_rate,
            groups=self.groups,
            base_width=self.base_width,
            depthwise=self.depthwise,
            extra=False,
        )
        return block

    def _transition(self, channels_in, channels_out):
        parts = []
        parts.append(GhostBatchNorm(channels_in, config["DataLoader"]["BatchSize"] // 32))
        parts.append(nn.CELU(alpha))
        parts.append(nn.Conv2d(channels_in, channels_out, kernel_size=1, stride=1))
        parts.append(nn.AvgPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*parts)
