import torch
import torch.nn as nn
import torch.nn.functional as F


def initial_conv(channels_in):
    return nn.Conv2d(3, channels_in, kernel_size=7, stride=2, padding=3, bias=False)


def conv_3x3(channels_in, channels_out, stride=1, groups=1, dilation=1):
    return nn.Conv2d(
        in_channels=channels_in,
        out_channels=channels_out,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        dilation=dilation,
        groups=groups,
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


class _base_block(nn.Module):
    expansion = 1

    def __init__(
        self, channels_in, channels_out, stride=1, groups=1, base_width=64, dilation=1,
    ):
        super().__init__()
        self.stride = 1
        self.bn1 = nn.BatchNorm2d(channels_in)
        self.conv1 = conv_3x3(channels_in, channels_out, stride)
        self.bn2 = nn.BatchNorm2d(channels_out)
        self.conv2 = conv_3x3(channels_out, channels_out)
        self.activation1 = nn.ReLU(inplace=True)
        self.activation2 = nn.ReLU(inplace=True)

        if groups != 1 or base_width != 64:
            raise ValueError("base_block only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in base_block")

    def _forward_impl(self, x):
        out = self.bn1(x)
        out = self.activation1(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.activation2(out)
        out = self.conv2(out)
        return out

    def forward(self, x):
        out = self._forward_impl(x)
        return out


class _bottleneck_block(nn.Module):
    expansion = 4

    def __init__(
        self, channels_in, channels_out, stride=1, groups=1, base_width=64, dilation=1
    ):
        super().__init__()
        width = int(channels_out * (base_width / 64.0)) * groups
        self.bn1 = nn.BatchNorm2d(channels_in)
        self.conv1 = conv_1x1(channels_in, width)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv2 = conv_3x3(
            width, width, stride=stride, groups=groups, dilation=dilation
        )
        self.bn3 = nn.BatchNorm2d(width)
        self.conv3 = conv_1x1(width, channels_out * self.expansion)
        self.activation1 = nn.ReLU(inplace=True)
        self.activation2 = nn.ReLU(inplace=True)
        self.activation3 = nn.ReLU(inplace=True)

    def _forward_impl(self, x):
        out = self.bn1(x)
        out = self.activation1(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.activation2(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.activation3(out)
        out = self.conv3(out)
        return out

    def forward(self, x):
        out = self._forward_impl(x)
        return out


class base_net(nn.Module):
    def __init__(
        self,
        name,
        block_type,
        layers,
        num_classes,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
    ):
        super().__init__()

        if isinstance(layers, int):
            layers = [1] * layers
        if replace_stride_with_dilation is None:
            self.replace_stride_with_dilation = [False] * (len(layers))

        if len(self.replace_stride_with_dilation) != len(layers):
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a n-element tuple, got {}".format(self.replace_stride_with_dilation)
            )

        self.name = name
        self.writer = ""
        self.num_classes = num_classes
        self.dilation = 1
        self.block_type = block_type
        self.groups = groups
        self.base_width = width_per_group
        self.block_type = block_type
        self.channels_in = 64
        self.conv0 = initial_conv(self.channels_in)
        self.bn0 = nn.BatchNorm2d(self.channels_in)
        self.activation0 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer_set = layers

        self.hidden_layers = self._build_layers(self.layer_set)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(
            self.block_type.expansion * 64 * (2 ** (len(layers) - 1)), num_classes
        )

        self._init_modules()

    def _init_modules(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for m in self.modules():
            if isinstance(m, _bottleneck_block):
                nn.init.constant_(m.bn3.weight, 0)
            elif isinstance(m, _base_block):
                nn.init.constant_(m.bn2.weight, 0)

    def _build_layers(self, layers):
        hidden_layers = [self._build_unit(64, layers[0])]
        for i in range(1, len(layers)):
            hidden_layers.append(
                self._build_unit(
                    64 * (2 ** i),
                    layers[i],
                    stride=2,
                    dilate=self.replace_stride_with_dilation[i],
                )
            )
        return nn.Sequential(*hidden_layers)

    def _build_unit(self, channels, blocks, stride=1, dilate=False):
        previous_dilation = self.dilation

        if dilate:
            self.dilation *= stride
            stride = 1

        layers = []
        layers.append(
            self.block_type(
                channels_in=self.channels_in,
                channels_out=channels,
                stride=stride,
                groups=self.groups,
                base_width=self.base_width,
                dilation=previous_dilation,
            )
        )
        self.channels_in = channels * self.block_type.expansion
        for _ in range(1, blocks):
            layers.append(
                self.block_type(
                    channels_in=self.channels_in,
                    channels_out=channels,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=previous_dilation,
                )
            )
        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        out = self.conv0(x)
        out = self.bn0(out)
        out = self.activation0(out)
        out = self.maxpool(out)

        out = self.hidden_layers(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out

    def forward(self, x):
        return self._forward_impl(x)


def _basenet(name, block_type, layers, num_classes=10):
    model = base_net(name, block_type, layers, num_classes)
    return model


def BaseTest():
    return _basenet("BaseTest", _base_block, [1, 1])


def BottleTest():
    return _basenet("BottleTest", _bottleneck_block, [1, 1])


def BaseNet34():
    return _basenet("BaseNet34", _base_block, [3, 4, 6, 3])


def BaseNet50():
    return _basenet("BaseNet50", _bottleneck_block, [3, 4, 6, 3])

