import torch
import torch.nn as nn
import torch.nn.functional as F
from model.common import *


class fractal_path(nn.Module):
    def __init__(
        self,
        block_type,
        channels_in,
        channels_out,
        fractal_expansion,
        distance,
        max_distance,
        stride=1,
        groups=1,
        base_width=64,
        dilation=1,
        prev_dilation=1,
        drop_path=None,
    ):
        super().__init__()
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.fractal_expansion = fractal_expansion
        self.distance = distance
        self.max_distance = max_distance
        self.drop_path = drop_path

        if distance == max_distance:
            self.conv1 = block_type(
                channels_in,
                channels_out,
                stride=stride,
                groups=groups,
                base_width=base_width,
                dilation=prev_dilation,
            )
        else:
            self.conv1 = block_type(
                channels_out * block_type.expansion,
                channels_out,
                groups=groups,
                base_width=base_width,
                dilation=dilation,
            )

        if fractal_expansion > 1:
            fractals = []
            fractals.append(
                _fractal_path(
                    block_type=block_type,
                    channels_in=channels_in,
                    channels_out=channels_out,
                    fractal_expansion=fractal_expansion - 1,
                    distance=distance,
                    max_distance=max_distance,
                    stride=stride,
                    groups=groups,
                    base_width=base_width,
                    dilation=dilation,
                    prev_dilation=prev_dilation,
                )
            )
            fractals.append(
                _fractal_path(
                    block_type=block_type,
                    channels_in=channels_in,
                    channels_out=channels_out,
                    fractal_expansion=self.fractal_expansion - 1,
                    distance=self.distance - 1,
                    max_distance=self.max_distance,
                    stride=stride,
                    groups=groups,
                    base_width=base_width,
                    dilation=dilation,
                    prev_dilation=prev_dilation,
                )
            )
            self.path = nn.Sequential(*fractals)

    def forward(self, x):
        if self.drop_path is not None:
            if self.drop_path == 0:
                prob = np.random.binomial(n=1, p=0.5)
            elif self.drop_path != self.fractal_expansion:
                prob = 0
            elif self.drop_path == self.fractal_expansion:
                prob = 1

            if prob == 1:
                if self.fractal_expansion == 1:
                    if x is None:
                        out = None
                    else:
                        out = self.conv1(x)
                else:
                    out_list = [self.conv1(x), self.path(x)]
                    if None in out_list:
                        out_list.remove(None)
                    out = torch.mean(torch.stack(out_list), dim=0)
            else:
                if self.fractal_expansion == 1:
                    out = None
                else:
                    out_list = [self.path(x)]
                    if None in out_list:
                        out_list.remove(None)
                    out = torch.mean(torch.stack(out_list), dim=0)

        else:
            if self.fractal_expansion == 1:
                out = self.conv1(x)
            else:
                out_list = [self.conv1(x), self.path(x)]
                out = torch.mean(torch.stack(out_list), dim=0)
        return out
