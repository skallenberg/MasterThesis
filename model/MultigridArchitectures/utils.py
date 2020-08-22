import torch
import torch.functional as F
import torch.nn as nn


class interpolate(nn.Module):
    def __init__(self, channels_in=None, scale=None, mode="nearest"):
        super().__init__()
        self.interpol = nn.functional.interpolate
        self.scale = scale
        self.mode = mode
        self.channels_in = channels_in
        if self.channels_in:
            self.conv = conv_1x1(
                channels_in=self.channels_in, channels_out=int(self.channels_in * self.scale),
            )

    def forward(self, x):
        out = self.interpol(x, scale_factor=self.scale, mode=self.mode)
        if self.channels_in:
            out = self.conv(out)

        return out


class transition(nn.Module):
    def __init__(self):
        super().__init__()
        parts = []
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=1)
        self.subsample = interpolate(scale=0.5)

    def forward(self, x):
        result = []

        for idx, input in enumerate(x):
            if idx < (len(x) - 1):
                if int(list(x[idx + 1].size())[2]) == 1 and (idx + 1) == (len(x) - 1):
                    result.append(torch.cat((self.subsample(self.pool(x[idx])), x[idx + 1]), 1))
                    break
                else:
                    result.append(self.subsample(self.pool(x[idx])))
            else:
                result.append(self.subsample(self.pool(x[idx])))
        return result
