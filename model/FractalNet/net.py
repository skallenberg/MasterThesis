import torch
import torch.nn as nn
import torch.nn.functional as F

from model.BaseNet.net import BaseNet
from model.common import *

from .blocks import *

import numpy as np

from utils.config import Config


class FractalNet(BaseNet):
    def __init__(
        self, name, block_type, layers, fractal_expansion=4, num_classes=10, drop_path=False,
    ):
        self.fractal_expansion = fractal_expansion
        self.drop_path = drop_path
        layers = [1] * layers
        super().__init__(
            name, block_type, layers, num_classes,
        )

        self.fc = nn.Linear(self.channels_in, self.num_classes)

    def _build_layers(self, layers):
        layers = len(layers)
        channels_in = self.channels_in
        hidden_layers = []

        for i in range(layers):
            if i == layers - 1:
                pool = False
            else:
                pool = True
            hidden_layers.append(self._build_unit(self.block_type, channels_in, stride=2,))
            channels_in = channels_in * 2 * self.block_type.expansion
            hidden_layers.append(self._transition(channels_in, pool=pool))

        self.channels_in = channels_in
        return nn.Sequential(*hidden_layers)

    def _build_unit(self, block_type, channels, stride=1):

        global_or_local = None
        if self.drop_path:
            global_or_local = np.random.binomial(n=1, p=0.5)
            if global_or_local == 1:
                rng = np.random.default_rng()
                global_or_local *= rng.integers(1, self.fractal_expansion + 1)

        block = fractal_path(
            block_type,
            channels,
            channels * 2,
            fractal_expansion=self.fractal_expansion,
            distance=2 ^ (self.fractal_expansion - 1),
            max_distance=2 ^ (self.fractal_expansion - 1),
            stride=stride,
            drop_path=global_or_local,
        )
        return block

    def _transition(self, channels_in, pool=True):
        if self.config["Misc"]["GhostBatchNorm"]:
            bn = GhostBatchNorm(channels_in, self.config["DataLoader"]["BatchSize"] // 32)
        else:
            bn = nn.BatchNorm2d(channels_in)

        if self.config["Misc"]["UseCELU"]:
            act = nn.CELU(self.config["Misc"]["CELU_alpha"])
        else:
            act = nn.ReLU(inplace=True)
        mpool = nn.MaxPool2d(2, stride=1, padding=1)

        return nn.Sequential(*[bn, act, mpool])
