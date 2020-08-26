from .blocks import *
from .net import ResNetv2


def _resnetv2(name, block_type, layers, num_classes=10, **kwargs):
    model = ResNetv2(name, block_type, layers, num_classes, **kwargs)
    return model


def ResBaseTestv2():
    return _resnetv2("ResBaseTest", residual_base_block, [1, 1])


def ResBottleTestv2():
    return _resnetv2("ResBottleTest", residual_bottleneck_block, [1, 1])


def StochasticDepthTestv2():
    return _resnetv2("ResBaseTest", residual_base_block, [1, 1], sd="prog")


def ResNet34v2():
    return _resnetv2("ResNet34", residual_base_block, [3, 4, 6, 3])


def ResNet50v2():
    return _resnetv2("ResNet50", residual_bottleneck_block, [3, 4, 6, 3])


def NeXtTestv2(**kwargs):
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 4
    return _resnetv2("NeXtTestv2", residual_bottleneck_block, [1, 1], **kwargs)


def DepthTestv2(**kwargs):
    kwargs["depthwise"] = True
    return _resnetv2("NeXtTestv2", residual_bottleneck_block, [1, 1], **kwargs)


def ResNeXt50v2(**kwargs):
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 4
    return _resnetv2("ResNeXt50v2", residual_bottleneck_block, [3, 4, 6, 3], **kwargs)


def WideTestv2(**kwargs):
    kwargs["width_per_group"] = 64 * 2
    return _resnetv2("WideTestv2", residual_bottleneck_block, [1, 1], **kwargs)


def WideResNet50v2(**kwargs):
    kwargs["width_per_group"] = 64 * 2
    return _resnetv2("WideResNet50v2", residual_bottleneck_block, [3, 4, 6, 3], **kwargs)
