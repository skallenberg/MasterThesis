from .blocks import *
from .net import BaseNetv2
from utils.config import Config

config = Config.get_instance()

if config["Setup"]["Data"] == "cifar100":
    num_classes = 100
else:
    num_classes = 10


def _basenetv2(name, block_type, layers, num_classes=num_classes, **kwargs):
    model = BaseNetv2(
        name=name, block_type=block_type, layers=layers, num_classes=num_classes, **kwargs
    )
    return model


def BaseTestv2(**kwargs):
    return _basenetv2("BaseTestv2", base_block, [1, 1, 1, 1], **kwargs)


def BaseTestv2NeXt(**kwargs):
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 4
    return _basenetv2("BaseTestv2", base_block, [1, 1, 1, 1], **kwargs)


def BaseTestv2Depth(**kwargs):
    kwargs["depthwise"] = True
    return _basenetv2("BaseTestv2", base_block, [1, 1, 1, 1], **kwargs)


def BottleTestv2(**kwargs):
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 4
    return _basenetv2("BottleTestv2", bottleneck_block, [1, 1], **kwargs)


def BaseNet34v2(**kwargs):
    return _basenetv2("BaseNet34v2", base_block, [3, 4, 6, 3])


def BaseNet50v2(**kwargs):
    return _basenetv2("BaseNet50v2", bottleneck_block, [3, 4, 6, 3])
