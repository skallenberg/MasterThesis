import torch
import torch.functional as F
import torch.nn as nn

from model.common import *


class interpolate(nn.Module):
    def __init__(self, channels_in=None, channel_scale=None, scale=None, mode="bilinear"):
        super().__init__()
        self.interpol = nn.functional.interpolate
        self.scale = scale
        self.mode = mode
        self.channels_in = channels_in
        self.channel_scale = channel_scale
        if self.channels_in:
            if self.channel_scale:
                channel_scale = self.channel_scale
            else:
                channel_scale = self.scale
            self.conv = conv_1x1(
                channels_in=self.channels_in, channels_out=int(self.channels_in * channel_scale),
            )

    def forward(self, x):
        out = self.interpol(x, scale_factor=self.scale, mode=self.mode)
        if self.channels_in:
            out = self.conv(out)

        return out
