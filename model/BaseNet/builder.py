from .blocks import *
from .net import BaseNet


def _basenet(name, block_type, layers, num_classes=10):
    model = BaseNet(name=name, block_type=block_type, layers=layers, num_classes=num_classes)
    return model


def BaseTest():
    return _basenet("BaseTest", base_block, [1, 1])


def BottleTest():
    return _basenet("BottleTest", bottleneck_block, [1, 1])


def BaseNet34():
    return _basenet("BaseNet34", base_block, [3, 4, 6, 3])


def BaseNet50():
    return _basenet("BaseNet50", bottleneck_block, [3, 4, 6, 3])
