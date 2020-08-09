import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.basenet import base_net, _base_block, _bottleneck_block


class _fractal_path(nn.Module):
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
                dilation=dilation,
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


class FractalNet(base_net):
    def __init__(
        self,
        name,
        block_type,
        layers,
        fractal_expansion=4,
        num_classes=10,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        drop_path=False,
    ):
        self.fractal_expansion = fractal_expansion
        self.drop_path = drop_path
        super().__init__(
            name,
            block_type,
            layers,
            num_classes,
            groups=groups,
            width_per_group=width_per_group,
            replace_stride_with_dilation=replace_stride_with_dilation,
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
            hidden_layers.append(
                self._build_unit(
                    self.block_type,
                    channels_in,
                    stride=2,
                    dilate=self.replace_stride_with_dilation[i],
                )
            )
            channels_in = channels_in * 2 * self.block_type.expansion
            hidden_layers.append(self._transition(channels_in, pool=pool))

        self.channels_in = channels_in
        return nn.Sequential(*hidden_layers)

    def _build_unit(self, block_type, channels, stride=1, dilate=False):

        previous_dilation = self.dilation

        global_or_local = None
        if self.drop_path:
            global_or_local = np.random.binomial(n=1, p=0.5)
            if global_or_local == 1:
                rng = np.random.default_rng()
                global_or_local *= rng.integers(1, self.fractal_expansion + 1)

        if dilate:
            self.dilation *= stride
            stride = 1

        block = _fractal_path(
            block_type,
            channels,
            channels * 2,
            fractal_expansion=self.fractal_expansion,
            distance=2 ^ (self.fractal_expansion - 1),
            max_distance=2 ^ (self.fractal_expansion - 1),
            stride=stride,
            groups=self.groups,
            base_width=self.base_width,
            dilation=previous_dilation,
            drop_path=global_or_local,
        )
        return block

    def _transition(self, channels_in, pool=True):
        parts = []
        parts.append(nn.BatchNorm2d(channels_in))
        parts.append(nn.ReLU(inplace=True))
        if pool:
            parts.append(nn.MaxPool2d(2, stride=1, padding=1))
        return nn.Sequential(*parts)


def _fractal_net(
    name, block_type, layers, fractal_expansion=4, num_classes=10, drop_path=False
):
    model = FractalNet(
        name,
        block_type,
        layers,
        fractal_expansion=fractal_expansion,
        num_classes=num_classes,
        drop_path=drop_path,
    )
    return model


def FractalTest():
    return _fractal_net("FractalTest", _bottleneck_block, 2, drop_path=True)


def FractalNet3():
    return _fractal_net("FractalNet3", _base_block, 3)


def FractalNet4():
    return _fractal_net("FractalNet4", _base_block, 4)


def FractalNet5():
    return _fractal_net("FractalNet5", _base_block, 5)
