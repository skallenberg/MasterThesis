import torch
import torch.nn as nn
import torch.nn.functional as F

from model.common import *
from utils.config import Config

from .utils import *


class mapping_block(nn.Module):
    def __init__(self, channels_in,channels_out, batch_norm=False):
        super().__init__()
        self.config = Config().get_instance()
        self.conv0 = self.conv0 = nn.Conv2d(channels_in, channels_out, kernel_size=3, stride=1, padding=1)
        if self.config["Misc"]["UseCELU"]:
            self.act0 = nn.CELU(self.config["Misc"]["CELU_alpha"])
        else:
            self.act0 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=1, padding=1)
        self.batch_norm = batch_norm
        if self.batch_norm:
            if self.config["Misc"]["GhostBatchNorm"]:
                self.bn0 = GhostBatchNorm(
                    channels_out, self.config["DataLoader"]["BatchSize"] // 32
                )
                self.bn1 = GhostBatchNorm(
                    channels_out, self.config["DataLoader"]["BatchSize"] // 32
                )
            else:
                self.bn0 = nn.BatchNorm2d(channels_out)
                self.bn1 = nn.BatchNorm2d(channels_out)

    def _forward_impl(self, x):
        if self.batch_norm:
            out = self.conv0(x)
            out = self.bn0(out)
            out = self.act0(out)
            out = self.conv1(out)
            out = self.bn1(out)
        else:
            out = self.conv0(x)
            #out = self.act0(out)
            #out = self.conv1(out)
        return out

    def forward(self, x):
        out = self._forward_impl(x)
        return out


class extractor_block(nn.Module):
    def __init__(self, channels_in, channels_out,  batch_norm=False):
        super().__init__()
        self.config = Config().get_instance()
        self.conv0 = nn.Conv2d(channels_in, channels_out, kernel_size=3, stride=1, padding=1)
        if self.config["Misc"]["UseCELU"]:
            self.act0 = nn.CELU(self.config["Misc"]["CELU_alpha"])
        else:
            self.act0 = nn.ReLU(inplace=True)
        self.batch_norm = batch_norm
        if self.batch_norm:
            if self.config["Misc"]["GhostBatchNorm"]:
                self.bn0 = GhostBatchNorm(
                    channels_out, self.config["DataLoader"]["BatchSize"] // 32
                )
            else:
                self.bn0 = nn.BatchNorm2d(channels_out)

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
