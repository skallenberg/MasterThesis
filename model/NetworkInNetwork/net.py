import torch
import torch.nn as nn
import torch.nn.functional as F

from model.BaseNet.net import BaseNet
from model.common import *

from .blocks import *


class NiN_Net(BaseNet):
    def __init__(
        self, name, block_type, layers, num_classes, full_connect=False,
    ):
        super().__init__(
            name, block_type, layers, num_classes,
        )
        self.full_connect = full_connect

    def _forward_impl(self, x):
        out = self.init_layer(x)
        out = self.maxpool(out)

        out = self.hidden_layers(out)

        if not self.full_connect:
            out = out.mean([2, 3])
        else:
            out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out
