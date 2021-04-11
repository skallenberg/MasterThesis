import torch
import torch.functional as F
import torch.nn as nn

from model.common import *


class interpolate(nn.Module):
    def __init__(self,channels_in=None, channel_scale=None, scale=None, mode="bilinear", type="interpol", in_size=None,):
        super().__init__()
        self.scale = scale
        self.mode = mode
        self.channels_in = channels_in
        self.channel_scale = channel_scale
        self.in_size = in_size
        if self.channels_in:
            if self.channel_scale:
                channel_scale = self.channel_scale
            else:
                channel_scale = self.scale
            self.conv = conv_1x1(
                channels_in=self.channels_in, channels_out=int(self.channels_in * channel_scale)
            )
        
        
        if type == "reflectpad":
            pad_size = int((in_size)/2)
            self.interpol = nn.ReflectionPad2d(pad_size)
        elif type == "replicpad":
            pad_size = int((in_size)/2)
            self.interpol = nn.ReplicationPad2d(pad_size)
        else:
            self.interpol = nn.Upsample(scale_factor=scale, mode=mode)

    def forward(self, x):
        out = self.interpol(x)
        
        if self.channels_in:
            out = self.conv(out)

        return out
