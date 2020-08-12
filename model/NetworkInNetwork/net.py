import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import *
from model.common import *
from model.BaseNet.net import BaseNet


class NiN_Net(BaseNet):
    def __init__(
        self,
        name,
        block_type,
        layers,
        num_classes,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        full_connect=False,
    ):
        super().__init__(
            name,
            block_type,
            layers,
            num_classes,
            groups=groups,
            width_per_group=width_per_group,
            replace_stride_with_dilation=replace_stride_with_dilation,
        )
        self.full_connect = full_connect

    def _forward_impl(self, x):
        out = self.conv0(x)
        out = self.bn0(out)
        out = self.activation0(out)
        out = self.maxpool(out)

        out = self.hidden_layers(out)

        if not self.full_connect:
            out = out.mean([2, 3])
        else:
            out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out
