from .blocks import *
from .net import BaseNetv2


def _basenetv2(name, block_type, layers, num_classes=10):
    model = BaseNetv2(name=name, block_type=block_type, layers=layers, num_classes=num_classes)
    return model


def BaseTestv2():
    return _basenetv2("BaseTestv2", base_block, [1, 2])


def BottleTestv2():
    return _basenetv2("BottleTestv2", bottleneck_block, [1, 1])


def BaseNet34v2():
    return _basenetv2("BaseNet34v2", base_block, [3, 4, 6, 3])


def BaseNet50v2():
    return _basenetv2("BaseNet50v2", bottleneck_block, [3, 4, 6, 3])
