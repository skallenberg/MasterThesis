from .blocks import *
from .net import ResNet


def _resnet(name, block_type, layers, num_classes=10, **kwargs):
    model = ResNet(name, block_type, layers, num_classes, **kwargs)
    return model


def ResBaseTest():
    return _resnet("ResBaseTest", residual_base_block, [1, 1])


def ResBottleTest():
    return _resnet("ResBottleTest", residual_bottleneck_block, [1, 1])


def StochasticDepthTest():
    return _resnet("ResBaseTest", residual_base_block, [1, 1], sd="prog")


def ResNet34():
    return _resnet("ResNet34", residual_base_block, [3, 4, 6, 3])


def ResNet50():
    return _resnet("ResNet50", residual_bottleneck_block, [3, 4, 6, 3])


def NeXtTest(**kwargs):
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 4
    return _resnet("NeXtTest", residual_bottleneck_block, [1, 1], **kwargs)


def ResNeXt50(**kwargs):
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 4
    return _resnet("ResNeXt50", residual_bottleneck_block, [3, 4, 6, 3], **kwargs)


def WideTest(**kwargs):
    kwargs["width_per_group"] = 64 * 2
    return _resnet("WideTest", residual_bottleneck_block, [1, 1], **kwargs)


def WideResNet50(**kwargs):
    kwargs["width_per_group"] = 64 * 2
    return _resnet("WideResNet50", residual_bottleneck_block, [3, 4, 6, 3], **kwargs)
