import torch
import torch.nn as nn
import torch.functional as F
from model.basenet import _base_block, _bottleneck_block


def initial_conv(channels_in):
    return nn.Conv2d(3, channels_in, kernel_size=7, stride=2, padding=3, bias=False)


def conv_3x3(channels_in, channels_out, stride=1):
    return nn.Conv2d(
        in_channels=channels_in,
        out_channels=channels_out,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def conv_1x1(channels_in, channels_out, stride=1):
    return nn.Conv2d(
        in_channels=channels_in,
        out_channels=channels_out,
        kernel_size=1,
        stride=stride,
        bias=False,
    )


class _interpolate(nn.Module):
    def __init__(self, channels_in=None, scale=None, mode="nearest"):
        super().__init__()
        self.interpol = nn.functional.interpolate
        self.scale = scale
        self.mode = mode
        self.channels_in = channels_in
        if self.channels_in:
            self.conv = conv_1x1(
                channels_in=self.channels_in,
                channels_out=int(self.channels_in * self.scale),
            )

    def forward(self, x):
        out = self.interpol(x, scale_factor=self.scale, mode=self.mode)
        if self.channels_in:
            out = self.conv(out)

        return out


class _mgconv_progressive_block(nn.Module):
    def __init__(
        self,
        block_type,
        channels_in,
        channels_out,
        stride=1,
        groups=1,
        base_width=64,
        dilation=1,
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

        if groups != 1 or base_width != 64:
            raise ValueError("base_block only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in base_block")

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
                    self.channels_out * self.block_type.expansion,
                    self.channels_out,
                    size=2,
                )
            )

        self.mg_conv_path_2 = nn.ModuleList()
        for i in range(self.nlayers):
            self.mg_conv_path_2.append(
                self._make_mg_path(
                    self.channels_out * self.block_type.expansion,
                    self.channels_out,
                    size=3,
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
            conv_1x1(
                self.channels_in // 2,
                self.channels_out // 2 * self.block_type.expansion,
            )
        )
        downsamples.append(
            conv_1x1(
                self.channels_in // 4,
                self.channels_out // 4 * self.block_type.expansion,
            )
        )

        return downsamples

    def _make_mg_path(self, channels_in, channels_out, size):
        mgconv_path = nn.ModuleList()
        if size == 3:
            mgconv_path.append(self.block_type(int(channels_in * 1.5), channels_out))
            mgconv_path.append(
                self.block_type(int(channels_in * 1.75), channels_out // 2)
            )
            mgconv_path.append(
                self.block_type(int(channels_in * 0.75), channels_out // 4)
            )
        elif size == 2:
            mgconv_path.append(
                self.block_type(int(channels_in * 0.75), int(channels_out * 0.5))
            )
            mgconv_path.append(
                self.block_type(int(channels_in * 0.75), int(channels_out * 0.25))
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


class _mgconv_base_block(nn.Module):
    def __init__(
        self,
        block_type,
        channels_in,
        channels_out,
        stride=1,
        groups=1,
        base_width=64,
        dilation=1,
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

        if groups != 1 or base_width != 64:
            raise ValueError("base_block only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in base_block")

        self.mgconv_path_1 = self._make_mg_path(
            self.channels_in, self.channels_out, size=self.size
        )

        self.mgconv_path_2 = self._make_mg_path(
            self.channels_out * self.block_type.expansion,
            self.channels_out,
            size=self.size,
        )

        if self.downsample_identity:
            self.downsample_list = self._make_downsamples()

    def _make_downsamples(self):
        downsamples = nn.ModuleList()
        if self.size == 3:
            downsamples.append(
                conv_1x1(
                    self.channels_in, self.channels_out * self.block_type.expansion
                )
            )
            downsamples.append(
                conv_1x1(
                    self.channels_in // 2,
                    self.channels_out // 2 * self.block_type.expansion,
                )
            )
            downsamples.append(
                conv_1x1(
                    self.channels_in // 4,
                    self.channels_out // 4 * self.block_type.expansion,
                )
            )
        elif self.size == 2:
            downsamples.append(
                conv_1x1(
                    int(self.channels_in),
                    self.channels_out * self.block_type.expansion,
                )
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
                self.block_type(int(channels_in * 1.75), channels_out // 2)
            )
            mgconv_path.append(
                self.block_type(int(channels_in * 0.75), channels_out // 4)
            )
        elif self.size == 2:
            mgconv_path.append(self.block_type(int(channels_in * 1.75), channels_out))
            mgconv_path.append(
                self.block_type(int(channels_in * 1.75), int(channels_out * 0.75))
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


class _transition(nn.Module):
    def __init__(self):
        super().__init__()
        parts = []
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=1)
        self.subsample = _interpolate(scale=0.5)

    def forward(self, x):
        result = []

        for idx, input in enumerate(x):
            if idx < (len(x) - 1):
                if int(list(x[idx + 1].size())[2]) == 1 and (idx + 1) == (len(x) - 1):
                    result.append(
                        torch.cat((self.subsample(self.pool(x[idx])), x[idx + 1]), 1)
                    )
                    break
                else:
                    result.append(self.subsample(self.pool(x[idx])))
            else:
                result.append(self.subsample(self.pool(x[idx])))
        return result


class MANNet(nn.Module):
    def __init__(
        self, name, block_type, layers, num_classes, residual=False, progressive=False
    ):
        super().__init__()
        self.name = name
        self.writer = ""
        self.num_classes = num_classes
        self.channels_in = 64
        self.layer_set = layers
        self.residual = residual
        self.progressive = progressive
        self.block_type = block_type
        self.conv0 = conv_3x3(3, self.channels_in)

        self.bn0 = nn.BatchNorm2d(self.channels_in)

        self.activation0 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.hidden_layers = self._build_layers(self.layer_set)

        self.fc = nn.Linear(int(self.channels_in * 1.75), self.num_classes)

        self.inital_downsample_1 = _interpolate(channels_in=64, scale=0.5)
        self.inital_downsample_2 = _interpolate(channels_in=64, scale=0.25)

    def _build_layers(self, layers):
        hidden_layers = []
        for i in range(len(layers) - 1):
            hidden_layers.append(
                self._build_unit(
                    self.block_type, 64 * (2 ** i), layers[i], downsample=True, size=3,
                )
            )
            hidden_layers.append(_transition())
        if self.block_type.expansion == 4:
            hidden_layers.append(
                self._build_unit(
                    self.block_type,
                    self.channels_in,
                    layers[-1],
                    downsample=True,
                    size=2,
                )
            )
        else:
            hidden_layers.append(
                self._build_unit(
                    self.block_type,
                    self.channels_in,
                    layers[-1],
                    downsample=False,
                    size=2,
                )
            )
        hidden_layers.append(_transition())
        if self.progressive:
            hidden_layers[0] = _mgconv_progressive_block(
                self.block_type, 64, 64, downsample_identity=True, nlayers=layers[0]
            )
        return nn.Sequential(*hidden_layers)

    def _build_unit(self, block_type, channels, nblocks, downsample, stride=1, size=3):
        layers = []
        layers.append(
            _mgconv_base_block(
                block_type=block_type,
                channels_in=self.channels_in,
                channels_out=channels,
                size=size,
                residual=self.residual,
                downsample_identity=downsample,
            )
        )
        self.channels_in = channels * self.block_type.expansion
        for _ in range(1, nblocks):
            layers.append(
                _mgconv_base_block(
                    block_type=block_type,
                    channels_in=self.channels_in,
                    channels_out=channels,
                    size=size,
                    residual=self.residual,
                )
            )
        return nn.Sequential(*layers)

    def _forward_impl(self, x):

        input = self.activation0(self.bn0(self.conv0(x)))

        out = [input] * 3
        out[1] = self.inital_downsample_1(input)
        out[2] = self.inital_downsample_2(input)

        out = self.hidden_layers(out)
        out = out[0]
        out = torch.flatten(out, 1)

        out = self.fc(out)

        return out

    def forward(self, x):
        out = self._forward_impl(x)
        return out


def _man_net(
    name, block_type, layers, num_classes=10, residual=False, progressive=False
):
    return MANNet(
        name,
        block_type,
        layers,
        num_classes,
        residual=residual,
        progressive=progressive,
    )


def MAN_Test():
    return _man_net(
        "MAN_Test", _base_block, [2, 2, 2, 2, 2], progressive=True, residual=True
    )


def MAN_Bottle_Test():
    return _man_net("MAN_Bottle_Test", _bottleneck_block, [2, 2, 2, 2, 2])


def MAN_Res_Test():
    return _man_net("MAN_Res_Test", _base_block, [2, 2, 2, 2, 2], residual=True)


def MAN_Res_Bottle_Test():
    return _man_net(
        "MAN_Res_Bottle_Test", _bottleneck_block, [2, 2, 2, 2, 2], residual=True
    )


def MG16():
    return _man_net("MG16", _base_block, [3, 4, 4, 4, 4],)


def R_MG16():
    return _man_net(
        "R_MG16", _base_block, [3, 4, 4, 4, 4], progressive=False, residual=True
    )


def PMG16():
    return _man_net(
        "PMG16", _base_block, [2, 3, 3, 3, 3], progressive=True, residual=False
    )


def R_PMG16():
    return _man_net(
        "R_PMG16", _base_block, [1, 2, 2, 2, 1], progressive=True, residual=True
    )
