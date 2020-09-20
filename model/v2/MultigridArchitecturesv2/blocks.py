import torch
import torch.nn as nn
import torch.nn.functional as F

from model.common import *
from utils.config import Config

from .utils import *

config = Config.get_instance()

alpha = config["Misc"]["CELU_alpha"]

apply = lambda x, y: x(y)


class mgconv_progressive_block(nn.Module):
    def __init__(
        self,
        block_type,
        channels_in,
        channels_out,
        stride=1,
        groups=1,
        base_width=64,
        residual=False,
        nlayers=2,
        depthwise=False,
        extra=False,
    ):
        super().__init__()
        self.stride = 1
        self.maxpool = nn.MaxPool2d(2, stride=1, padding=1)
        self.upsample = interpolate(scale=2)
        self.downsample = interpolate(scale=0.5)
        self.downsample_list = None
        self.nlayers = nlayers
        self.residual = residual
        self.block_type = block_type
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.groups = groups
        self.base_width = base_width
        self.depthwise = depthwise
        self.extra = extra

        inital_conv = nn.ModuleList()
        ch_in = self.channels_in
        ch_out = self.channels_out

        if self.depthwise:
            groups = ch_in // 4

        parts = []
        for i in range(self.nlayers):
            parts.append(
                conv_3x3(ch_in // 4, ch_out // 4 * self.block_type.expansion, groups=groups)
            )
            parts.append(
                GhostBatchNorm(
                    ch_in // 4 * self.block_type.expansion, config["DataLoader"]["BatchSize"] // 32
                )
            )
            parts.append(nn.CELU(alpha))
            ch_in = ch_out * self.block_type.expansion
            groups = ch_in // 4
        self.initial_conv = nn.Sequential(*parts)

        self.mg_conv_path_1 = nn.ModuleList(
            [
                self._make_mg_path(
                    self.channels_out * self.block_type.expansion, self.channels_out, size=2,
                )
                for i in range(self.nlayers)
            ]
        )

        self.mg_conv_path_2 = nn.ModuleList(
            [
                self._make_mg_path(
                    self.channels_out * self.block_type.expansion, self.channels_out, size=3,
                )
                for i in range(self.nlayers)
            ]
        )

        self.downsample_list = self._make_downsamples()

    def _make_downsamples(self):
        downsamples = nn.ModuleList()
        downsamples.append(
            conv_1x1(self.channels_in, self.channels_out * self.block_type.expansion)
        )
        downsamples.append(
            conv_1x1(self.channels_in // 2, self.channels_out // 2 * self.block_type.expansion,)
        )
        downsamples.append(
            conv_1x1(self.channels_in // 4, self.channels_out // 4 * self.block_type.expansion,)
        )

        return downsamples

    def _make_mg_path(self, channels_in, channels_out, size):
        mgconv_path = nn.ModuleList()
        if size == 3:
            mgconv_path.append(self.block_type(int(channels_in * 1.5), channels_out))
            mgconv_path.append(
                self.block_type(
                    channels_in=int(channels_in * 1.75),
                    channels_out=channels_out // 2,
                    groups=self.groups,
                    base_width=self.base_width,
                    depthwise=self.depthwise,
                    extra=self.extra,
                )
            )
            mgconv_path.append(
                self.block_type(
                    channels_in=int(channels_in * 0.75),
                    channels_out=channels_out // 4,
                    groups=self.groups,
                    base_width=self.base_width,
                    depthwise=self.depthwise,
                    extra=self.extra,
                )
            )
        elif size == 2:
            mgconv_path.append(
                self.block_type(
                    channels_in=int(channels_in * 0.75),
                    channels_out=int(channels_out * 0.5),
                    groups=self.groups,
                    base_width=self.base_width,
                )
            )
            mgconv_path.append(
                self.block_type(
                    channels_in=int(channels_in * 0.75),
                    channels_out=int(channels_out * 0.25),
                    groups=self.groups,
                    base_width=self.base_width,
                    depthwise=self.depthwise,
                    extra=self.extra,
                )
            )
        return mgconv_path

    def _concat(self, x):
        """grids = []

        for idx in range(len(x)):
            neighbours = []
            if idx == 0:
                if len(x) > 1:
                    neighbours.append(self.upsample(x[idx + 1]))
                neighbours.append(x[idx])
            if idx == 1:
                if len(x) > 2:
                    neighbours.append(self.upsample(x[idx + 1]))
                neighbours.append(x[idx])
                neighbours.append(self.downsample(x[idx - 1]))
            if idx == 2:
                neighbours.append(self.downsample(x[idx - 1]))
                neighbours.append(x[idx])  
            grids.append(torch.cat(neighbours, 1))"""

        n = len(x)
        if n == 1:
            pairs = [[x[0]]]
        elif n == 2:
            pairs = [[self.upsample(x[1]), x[0]], [x[1], self.downsample(x[0])]]
        elif n == 3:
            pairs = [
                [self.upsample(x[1]), x[0]],
                [self.upsample(x[2]), x[1], self.downsample(x[0])],
                [self.downsample(x[1]), x[2]],
            ]
        cat_ = lambda x: torch.cat(x, 1)
        grids = map(cat_, pairs)
        return grids

    def _forward_impl(self, x):
        # Input should be made out of 1-3 "grids" so  3 separate inputs
        # Input is combined with its neighbours through up and downsampling
        results = []

        if self.residual:
            out = x[2]
            identity = self.downsample_list[2](out)
            for idx in range(self.nlayers):
                out = self.initial_conv[idx](out)
            out += identity

            n_elem = self.downsample_list[1](x[1])
            out = [n_elem, out]
            identity = out
            for idx in range(self.nlayers):
                out = self._concat(out)
                out = [self.mg_conv_path_1[idx][i](out) for i in range(2)]
                out = [out[i] + identity[i] for i in range(2)]

            out = [self.downsample_list[0](x[0]), out]
            identity = out
            for idx in range(self.nlayers):
                out = self._concat(out)
                out = [self.mg_conv_path_2[idx][i](out) for i in range(3)]
            out = [out[i] + identity[i] for i in range(3)]

        else:
            out = self.initial_conv(x[2])

            out = [self.downsample_list[1](x[1]), out]
            for idx in range(self.nlayers):
                out = self._concat(out)
                result = map(apply, self.mg_conv_path_1[idx], out)
                out = result
                # out = [self.mg_conv_path_1[idx][i](out[i]) for i in range(2)]

            out = [self.downsample_list[0](x[0]), *out]
            for idx in range(self.nlayers):
                out = self._concat(out)
                out = map(apply, self.mg_conv_path_2[idx], out)
                # out = [self.mg_conv_path_2[idx][i](out[i]) for i in range(3)]
        return list(out)

    def forward(self, x):

        out = self._forward_impl(x)

        return out


class mgconv_base_block(nn.Module):
    def __init__(
        self,
        block_type,
        channels_in,
        channels_out,
        stride=1,
        groups=1,
        base_width=64,
        size=3,
        residual=False,
        downsample_identity=False,
        depthwise=False,
        extra=False,
    ):
        super().__init__()
        self.stride = 1
        self.maxpool = nn.MaxPool2d(2, stride=1, padding=1)
        self.upsample = interpolate(scale=2)
        self.downsample = interpolate(scale=0.5)
        self.downsample_identity = downsample_identity
        self.downsample_list = None
        self.size = size
        self.residual = residual
        self.block_type = block_type
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.groups = groups
        self.base_width = base_width
        self.depthwise = depthwise
        self.extra = extra

        self.mgconv_path_1 = self._make_mg_path(
            self.channels_in, self.channels_out, size=self.size
        )

        if self.downsample_identity:
            self.downsample_list = self._make_downsamples()

    def _make_downsamples(self):
        downsamples = nn.ModuleList()
        if self.size == 3:
            downsamples.append(
                conv_1x1(self.channels_in, self.channels_out * self.block_type.expansion)
            )
            downsamples.append(
                conv_1x1(
                    self.channels_in // 2, self.channels_out // 2 * self.block_type.expansion,
                )
            )
            downsamples.append(
                conv_1x1(
                    self.channels_in // 4, self.channels_out // 4 * self.block_type.expansion,
                )
            )
        elif self.size == 2:
            downsamples.append(
                conv_1x1(int(self.channels_in), self.channels_out * self.block_type.expansion,)
            )
            downsamples.append(
                conv_1x1(
                    int(self.channels_in * 0.75),
                    int(self.channels_out * 0.75) * self.block_type.expansion,
                )
            )
        return downsamples

    def _make_mg_path(self, channels_in, channels_out, size):
        mgconv_path = nn.ModuleList()
        if self.size == 3:
            mgconv_path.append(
                self.block_type(
                    int(channels_in * 1.5),
                    channels_out,
                    groups=self.groups,
                    base_width=self.base_width,
                    depthwise=self.depthwise,
                    extra=self.extra,
                )
            )
            mgconv_path.append(
                self.block_type(
                    int(channels_in * 1.75),
                    channels_out // 2,
                    groups=self.groups,
                    base_width=self.base_width,
                    depthwise=self.depthwise,
                    extra=self.extra,
                )
            )
            mgconv_path.append(
                self.block_type(
                    int(channels_in * 0.75),
                    channels_out // 4,
                    groups=self.groups,
                    base_width=self.base_width,
                    depthwise=self.depthwise,
                    extra=self.extra,
                )
            )
        elif self.size == 2:
            mgconv_path.append(
                self.block_type(
                    int(channels_in * 1.75),
                    channels_out,
                    groups=self.groups,
                    base_width=self.base_width,
                    depthwise=self.depthwise,
                    extra=self.extra,
                )
            )
            mgconv_path.append(
                self.block_type(
                    int(channels_in * 1.75),
                    int(channels_out * 0.75),
                    groups=self.groups,
                    base_width=self.base_width,
                    depthwise=self.depthwise,
                    extra=self.extra,
                )
            )
        return mgconv_path

    def _concat(self, x):
        """grids = []

        for idx in range(len(x)):
            neighbours = []
            if idx == 0:
                if len(x) > 1:
                    neighbours.append(self.upsample(x[idx + 1]))
                neighbours.append(x[idx])
            if idx == 1:
                if len(x) > 2:
                    neighbours.append(self.upsample(x[idx + 1]))
                neighbours.append(x[idx])
                neighbours.append(self.downsample(x[idx - 1]))
            if idx == 2:
                neighbours.append(self.downsample(x[idx - 1]))
                neighbours.append(x[idx])  
            grids.append(torch.cat(neighbours, 1))"""

        n = len(x)
        if n == 1:
            pairs = [[x[0]]]
        elif n == 2:
            pairs = [[self.upsample(x[1]), x[0]], [x[1], self.downsample(x[0])]]
        elif n == 3:
            pairs = [
                [self.upsample(x[1]), x[0]],
                [self.upsample(x[2]), x[1], self.downsample(x[0])],
                [self.downsample(x[1]), x[2]],
            ]
        cat_ = lambda x: torch.cat(x, 1)
        grids = map(cat_, pairs)
        return grids

    def _forward_impl(self, x):
        # Input should be made out of 1-3 "grids" so  3 separate inputs
        # Input is combined with its neighbours through up and downsampling
        # results = []
        grids = self._concat(x)

        # for idx, input in enumerate(grids):
        #    out = self.mgconv_path_1[idx](input)
        #    results.append(out)

        results = list(map(apply, self.mgconv_path_1, grids))

        return results

    def forward(self, x):
        if self.residual:
            identity = x
            out = self._forward_impl(x)
            for idx in range(len(x)):
                if self.downsample_list:
                    downsampled_identity = self.downsample_list[idx](identity[idx])
                else:
                    downsampled_identity = identity[idx]

                out[idx] += downsampled_identity
        else:
            out = self._forward_impl(x)

        return out
