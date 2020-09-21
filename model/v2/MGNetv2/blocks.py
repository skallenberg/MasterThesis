import torch
import torch.nn as nn
import torch.nn.functional as F

from model.common import *
from utils.config import Config

from .utils import *

config = Config.get_instance()

alpha = config["Misc"]["CELU_alpha"]


class mapping_block(nn.Module):
    def __init__(self, channels_in, batch_norm=False):
        super().__init__()
        self.conv0 = conv_3x3(channels_in, channels_in)
        self.act0 = nn.CELU(alpha)
        self.conv1 = conv_3x3(channels_in, channels_in)
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.bn0 = GhostBatchNorm(channels_in, config["DataLoader"]["BatchSize"] // 32)
            self.bn1 = GhostBatchNorm(channels_in, config["DataLoader"]["BatchSize"] // 32)

    def _forward_impl(self, x):
        if self.batch_norm:
            out = self.conv0(x)
            out = self.bn0(out)
            out = self.act0(out)
            out = self.conv1(out)
            out = self.bn1(out)
        else:
            out = self.conv0(x)
            out = self.act0(out)
            out = self.conv1(out)
        return out

    def forward(self, x):
        out = self._forward_impl(x)
        return out


class extractor_block(nn.Module):
    def __init__(self, channels_in, batch_norm=False):
        super().__init__()
        self.conv0 = conv_3x3(channels_in, channels_in)
        self.act0 = nn.CELU(alpha)
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.bn0 = GhostBatchNorm(channels_in, config["DataLoader"]["BatchSize"] // 32)

    def _forward_impl(self, x):
        if self.batch_norm:
            out = self.act0(x)
            out = self.conv0(out)
            out = self.bn0(out)
            out = self.act0(out)
        else:
            out = self.act0(x)
            out = self.conv0(out)
            out = self.act0(out)
        return out

    def forward(self, x):
        out = self._forward_impl(x)
        return out
