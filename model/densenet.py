import torch
import torch.nn as nn
import torch.nn.functional as F
from model.basenet import base_net, _base_block, _bottleneck_block


class _dense_base_unit(_base_block):
    expansion = 1

    def __init__(
        self,
        channels_in,
        channels_out,
        drop_rate,
        stride=1,
        groups=1,
        base_width=64,
        dilation=1,
    ):
        super().__init__(
            channels_in,
            channels_out,
            stride=stride,
            groups=groups,
            base_width=base_width,
            dilation=dilation,
        )
        self.drop_rate = float(drop_rate)

    def forward(self, x):

        out = torch.cat(x, 1)

        out = self._forward_impl(out)

        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)

        return out


class _dense_orig_bottleneck_unit(_dense_base_unit):
    expansion = 4

    def __init__(
        self,
        channels_in,
        channels_out,
        drop_rate,
        stride=1,
        groups=1,
        base_width=64,
        dilation=1,
    ):
        super().__init(
            channels_in,
            channels_out * self.expansion,
            drop_rate,
            stride=stride,
            groups=groups,
            base_width=base_width,
            dilation=dilation,
        )


class _dense_alternative_bottleneck_unit(_bottleneck_block):
    expansion = 4

    def __init__(
        self,
        channels_in,
        channels_out,
        drop_rate,
        stride=1,
        groups=1,
        base_width=64,
        dilation=1,
    ):
        super().__init(
            channels_in,
            channels_out,
            stride=stride,
            groups=groups,
            base_width=base_width,
            dilation=dilation,
        )

        self.drop_rate = float(drop_rate)

    def forward(self, x):

        out = torch.cat(x, 1)
        out = self._forward_impl(out)
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)

        return out


class _dense_block(nn.ModuleDict):
    def __init__(
        self,
        block_type,
        nlayers,
        channels_in,
        growth_rate,
        drop_rate,
        stride=1,
        groups=1,
        base_width=64,
        dilation=1,
    ):
        super().__init__()
        layer = block_type(
            channels_in=channels_in,
            channels_out=growth_rate,
            drop_rate=drop_rate,
            stride=stride,
            groups=groups,
            base_width=base_width,
            dilation=dilation,
        )
        self.add_module("denseblock_1", layer)
        for i in range(1, nlayers):
            layer = block_type(
                channels_in=channels_in + i * growth_rate,
                channels_out=growth_rate,
                drop_rate=drop_rate,
                groups=groups,
                base_width=base_width,
                dilation=dilation,
            )
            self.add_module("denseblock_%d" % (i + 1), layer)

    def forward(self, x):
        out = [x]
        for name, layer in self.items():
            out_append = layer(out)
            out.append(out_append)

        return torch.cat(out, 1)


class DenseNet(base_net):
    def __init__(
        self,
        name,
        block_type,
        layers,
        drop_rate=0,
        num_classes=10,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
    ):
        self.growth_rate = 32
        self.drop_rate = drop_rate
        super().__init__(
            name,
            block_type,
            layers,
            num_classes,
            groups=groups,
            width_per_group=width_per_group,
            replace_stride_with_dilation=replace_stride_with_dilation,
        )

        self.fc = nn.Linear(self.nfeats, self.num_classes)

    def _build_layers(self, layers):

        nfeats = self.channels_in
        hidden_layers = []
        for i in range(len(layers)):
            hidden_layers.append(
                self._build_unit(
                    self.block_type,
                    layers[i],
                    nfeats,
                    stride=1,
                    dilate=self.replace_stride_with_dilation[i],
                )
            )
            nfeats += layers[i] * self.growth_rate
            if i != len(layers) - 1:
                hidden_layers.append(self._transition(nfeats, int(nfeats / 2)))
                nfeats = int(nfeats / 2)

        self.bnf = nn.BatchNorm2d(nfeats)
        self.activationf = nn.ReLU(inplace=True)

        hidden_layers.append(self.bnf)
        hidden_layers.append(self.activationf)

        self.nfeats = nfeats

        self.hidden_layers = nn.Sequential(*hidden_layers)
        return nn.Sequential(*hidden_layers)

    def _build_unit(self, block_type, layer, channels, stride=1, dilate=False):
        previous_dilation = self.dilation

        if dilate:
            self.dilation *= stride
            stride = 1

        block = _dense_block(
            block_type,
            layer,
            channels,
            growth_rate=self.growth_rate,
            drop_rate=self.drop_rate,
            stride=stride,
            groups=self.groups,
            base_width=self.base_width,
            dilation=previous_dilation,
        )
        return block

    def _transition(self, channels_in, channels_out):
        parts = []
        parts.append(nn.BatchNorm2d(channels_in))
        parts.append(nn.ReLU(inplace=True))
        parts.append(nn.Conv2d(channels_in, channels_out, kernel_size=1, stride=1))
        parts.append(nn.AvgPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*parts)


def _densenet(name, block_type, layers, num_classes=10):
    model = DenseNet(name, block_type, layers, num_classes=num_classes)
    return model


def DenseTest():
    return _densenet("DenseTest", _dense_base_unit, [2, 2])


def DenseNet34():
    return _densenet("DenseNet34", _dense_base_unit, [3, 4, 6, 3])


def DenseNet50():
    return _densenet("DenseNet50", _dense_alternative_bottleneck_unit, [3, 4, 6, 3])


def DenseNet50_2():
    return _densenet("DenseNet50", _dense_orig_bottleneck_unit, [3, 4, 6, 3])


def DenseNet121():
    return _densenet("DenseNet121", _dense_orig_bottleneck_unit, [6, 12, 12, 16])


def DenseNet161():
    return _densenet("DenseNet161", _dense_orig_bottleneck_unit, [6, 12, 36, 24])


def DenseNet169():
    return _densenet("DenseNet169", _dense_orig_bottleneck_unit, [6, 12, 32, 32])


def DenseNet201():
    return _densenet("DenseNet201", _dense_orig_bottleneck_unit, [6, 12, 48, 32])

