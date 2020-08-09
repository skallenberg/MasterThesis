import torch
import torch.nn as nn
import torch.nn.functional as F
from model.resnet import ResNet, _residual_base_block, _residual_bottleneck_block


class RoR_Net(ResNet):
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
        super().__init__(
            name,
            block_type,
            layers,
            num_classes,
            groups=groups,
            width_per_group=width_per_group,
            replace_stride_with_dilation=replace_stride_with_dilation,
        )

        self.downsample_0 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.Conv2d(
                64,
                self.block_type.expansion * 64 * (2 ** (len(layers) - 1)),
                kernel_size=1,
                stride=2,
                bias=False,
            ),
        )

    def _build_layers(self, layers):
        hidden_layers = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        hidden_layers.append(self._build_unit(64, layers[0]))
        for i in range(1, len(layers)):
            hidden_layers.append(
                self._build_unit(
                    64 * (2 ** i),
                    layers[i],
                    stride=2,
                    dilate=self.replace_stride_with_dilation[i],
                )
            )
        channels_in = 64
        self.downsample_layers.append(
            nn.Sequential(
                nn.BatchNorm2d(channels_in),
                nn.Conv2d(
                    channels_in,
                    64 * self.block_type.expansion,
                    kernel_size=1,
                    stride=1,
                    bias=False,
                ),
            )
        )
        for i in range(1, len(layers)):
            channels = 64 * (2 ** i)
            self.downsample_layers.append(
                nn.Sequential(
                    nn.BatchNorm2d(channels_in),
                    nn.Conv2d(
                        channels_in,
                        channels * self.block_type.expansion,
                        kernel_size=1,
                        stride=2,
                        bias=False,
                    ),
                )
            )
            channels_in = channels * self.block_type.expansion

        return hidden_layers

    def _forward_impl(self, x):
        out = self.conv0(x)

        out = self.bn0(out)
        out = self.activation0(out)
        out = self.maxpool(out)

        identity_0 = out

        for idx, layer in enumerate(self.hidden_layers):
            identity_x = out
            out = layer(out)
            identity_x = self.downsample_layers[idx](identity_x)
            out = out + identity_x

        identity_0 = self.downsample_0(identity_0)
        out = out + identity_0
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


def _ror_net(name, block_type, layers, num_classes=10, **kwargs):
    model = RoR_Net(name, block_type, layers, num_classes, **kwargs)
    return model


def RoR_BaseTest():
    return _ror_net("RoR_BaseTest", _residual_base_block, [1, 1])


def RoR_BottleTest():
    return _ror_net("RoR_BottleTest", _residual_bottleneck_block, [1, 1])


def RoR_Net34():
    return _ror_net("RoR_Net34", _residual_base_block, [3, 4, 6, 3])


def RoR_Net50():
    return _ror_net("RoR_Net50", _residual_bottleneck_block, [3, 4, 6, 3])


def RoR_NeXt50(**kwargs):
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 4
    return _ror_net("RoR_NeXt50", _residual_bottleneck_block, [3, 4, 6, 3], **kwargs)


def RoR_WideResNet50(**kwargs):
    kwargs["width_per_group"] = 64 * 2
    return _ror_net(
        "RoR_WideResNet50", _residual_bottleneck_block, [3, 4, 6, 3], **kwargs
    )

