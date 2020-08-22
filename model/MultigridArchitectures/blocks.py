import torch
import torch.nn as nn
import torch.nn.functional as F

from model.common import *

from .utils import *


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
        downsample_identity=False,
        nlayers=2,
    ):
        super().__init__()
        self.stride = 1
        self.maxpool = nn.MaxPool2d(2, stride=1, padding=1)
        self.upsample = _interpolate(scale=2)
        self.downsample = _interpolate(scale=0.5)
        self.downsample_identity = downsample_identity
        self.downsample_list = None
        self.nlayers = nlayers
        self.residual = residual
        self.block_type = block_type
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.groups = groups
        self.base_width = base_width

        if self.block_type.expansion == 1:
            if groups != 1 or base_width != 64:
                raise ValueError("base_block only supports groups=1 and base_width=64")

        inital_conv = nn.ModuleList()
        ch_in = self.channels_in
        ch_out = self.channels_out
        for i in range(self.nlayers):
            parts = []
            parts.append(nn.ReLU(inplace=True))
            parts.append(nn.BatchNorm2d(ch_in // 4))
            parts.append(conv_3x3(ch_in // 4, ch_out // 4 * self.block_type.expansion))
            inital_conv.append(nn.Sequential(*parts))
            ch_in = ch_out * self.block_type.expansion
        self.initial_conv = inital_conv

        self.mg_conv_path_1 = nn.ModuleList()
        for i in range(self.nlayers):
            self.mg_conv_path_1.append(
                self._make_mg_path(
                    self.channels_out * self.block_type.expansion, self.channels_out, size=2,
                )
            )

        self.mg_conv_path_2 = nn.ModuleList()
        for i in range(self.nlayers):
            self.mg_conv_path_2.append(
                self._make_mg_path(
                    self.channels_out * self.block_type.expansion, self.channels_out, size=3,
                )
            )

        if self.downsample_identity:
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
                )
            )
            mgconv_path.append(
                self.block_type(
                    channels_in=int(channels_in * 0.75),
                    channels_out=channels_out // 4,
                    groups=self.groups,
                    base_width=self.base_width,
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
                )
            )
        return mgconv_path

    def _concat(self, x):
        grids = []

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
            grids.append(torch.cat(neighbours, 1))
        return grids

    def _forward_impl(self, x):
        # Input should be made out of 1-3 "grids" so  3 separate inputs
        # Input is combined with its neighbours through up and downsampling
        results = []

        if self.residual:
            out = x[2]
            for idx in range(self.nlayers):
                identity = out
                if idx == 0:
                    identity = self.downsample_list[2](identity)
                out = self.initial_conv[idx](out)
                out += identity

            identity = [self.downsample_list[1](x[1]), out]
            out = [self.downsample_list[1](x[1]), out]
            for idx in range(self.nlayers):
                out = self._concat(out)
                out = [self.mg_conv_path_1[idx][i](out) for i in range(2)]
                out = [out[i] + identity[i] for i in range(2)]

            new_elem = [self.downsample_list[0](x[0])]
            new_elem.extend(out)
            out = new_elem
            identity = new_elem
            for idx in range(self.nlayers):
                out = self._concat(out)
                out = [self.mg_conv_path_2[idx][i](out) for i in range(3)]
                out = [out[i] + identity[i] for i in range(3)]

        else:
            out = x[2]
            for idx in range(len(self.initial_conv)):
                out = self.initial_conv[idx](out)

            out = [self.downsample_list[1](x[1]), out]
            for idx in range(self.nlayers):
                out = self._concat(out)
                out = [self.mg_conv_path_1[idx][i](out[i]) for i in range(2)]

            new_elem = [self.downsample_list[0](x[0])]
            new_elem.extend(out)
            out = new_elem
            for idx in range(self.nlayers):
                out = self._concat(out)
                out = [self.mg_conv_path_2[idx][i](out[i]) for i in range(3)]
        return out

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
    ):
        super().__init__()
        self.stride = 1
        self.maxpool = nn.MaxPool2d(2, stride=1, padding=1)
        self.upsample = _interpolate(scale=2)
        self.downsample = _interpolate(scale=0.5)
        self.downsample_identity = downsample_identity
        self.downsample_list = None
        self.size = size
        self.residual = residual
        self.block_type = block_type
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.groups = groups
        self.base_width = base_width

        if self.block_type.expansion == 1:
            if groups != 1 or base_width != 64:
                raise ValueError("base_block only supports groups=1 and base_width=64")

        self.mgconv_path_1 = self._make_mg_path(
            self.channels_in, self.channels_out, size=self.size
        )

        self.mgconv_path_2 = self._make_mg_path(
            self.channels_out * self.block_type.expansion, self.channels_out, size=self.size,
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
            mgconv_path.append(self.block_type(int(channels_in * 1.5), channels_out))
            mgconv_path.append(
                self.block_type(
                    int(channels_in * 1.75),
                    channels_out // 2,
                    groups=self.groups,
                    base_width=self.base_width,
                )
            )
            mgconv_path.append(
                self.block_type(
                    int(channels_in * 0.75),
                    channels_out // 4,
                    groups=self.groups,
                    base_width=self.base_width,
                )
            )
        elif self.size == 2:
            mgconv_path.append(
                self.block_type(
                    int(channels_in * 1.75),
                    channels_out,
                    groups=self.groups,
                    base_width=self.base_width,
                )
            )
            mgconv_path.append(
                self.block_type(
                    int(channels_in * 1.75),
                    int(channels_out * 0.75),
                    groups=self.groups,
                    base_width=self.base_width,
                )
            )
        return mgconv_path

    def _concat(self, x):
        grids = []

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
            grids.append(torch.cat(neighbours, 1))
        return grids

    def _forward_impl(self, x):
        # Input should be made out of 1-3 "grids" so  3 separate inputs
        # Input is combined with its neighbours through up and downsampling
        results = []
        results_2 = []
        grids_1 = self._concat(x)

        for idx, input in enumerate(grids_1):
            out = self.mgconv_path_1[idx](input)
            results.append(out)

        # grids_2 = self._concat(results)

        # for idx, input in enumerate(grids_2):
        #    out = self.mgconv_path_2[idx](input)
        #    results_2.append(out)
        #    results = results_2

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
