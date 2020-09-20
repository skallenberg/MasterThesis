import torch
import torch.nn as nn
import torch.nn.functional as F

from model.BaseNetv2.blocks import *
from model.common import *


class dense_base_unit(base_block):
    expansion = 1

    def __init__(
        self,
        channels_in,
        channels_out,
        drop_rate,
        groups=1,
        base_width=64,
        depthwise=False,
        extra=False,
    ):
        super().__init__(
            channels_in,
            channels_out,
            groups=groups,
            base_width=base_width,
            depthwise=depthwise,
            extra=extra,
        )
        self.drop_rate = float(drop_rate)

    def forward(self, x):

        out = torch.cat(x, 1)

        out = self._forward_impl(out)

        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)

        return out


class dense_orig_bottleneck_unit(dense_base_unit):
    expansion = 4

    def __init__(
        self,
        channels_in,
        channels_out,
        drop_rate,
        groups=1,
        base_width=64,
        depthwise=False,
        extra=False,
    ):
        super().__init__(
            channels_in,
            channels_out * self.expansion,
            drop_rate,
            groups=groups,
            base_width=base_width,
            depthwise=depthwise,
            extra=extra,
        )


class dense_alternative_bottleneck_unit(bottleneck_block):
    expansion = 4

    def __init__(
        self,
        channels_in,
        channels_out,
        drop_rate,
        groups=1,
        base_width=64,
        depthwise=False,
        extra=False,
    ):
        super().__init__(
            channels_in,
            channels_out,
            groups=groups,
            base_width=base_width,
            depthwise=depthwise,
            extra=extra,
        )

        self.drop_rate = float(drop_rate)

    def forward(self, x):

        out = torch.cat(x, 1)
        out = self._forward_impl(out)
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)

        return out


class dense_block(nn.ModuleDict):
    def __init__(
        self,
        block_type,
        nlayers,
        channels_in,
        growth_rate,
        drop_rate,
        groups=1,
        base_width=64,
        depthwise=False,
        extra=False,
    ):
        super().__init__()
        layer = block_type(
            channels_in=channels_in,
            channels_out=growth_rate,
            drop_rate=drop_rate,
            groups=groups,
            base_width=base_width,
            depthwise=depthwise,
            extra=extra,
        )
        self.add_module("denseblock_1", layer)
        for i in range(1, nlayers):
            layer = block_type(
                channels_in=channels_in + i * growth_rate * block_type.expansion,
                channels_out=growth_rate,
                drop_rate=drop_rate,
                groups=groups,
                base_width=base_width,
                depthwise=depthwise,
                extra=extra,
            )
            self.add_module("denseblock_%d" % (i + 1), layer)

    def forward(self, x):
        out = [x]
        for name, layer in self.items():
            out_append = layer(out)
            out.append(out_append)

        return torch.cat(out, 1)
