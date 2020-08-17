from .net import BaseNet
from .blocks import *


def _basenet(name, block_type, layers, num_classes=10):
    model = BaseNet(name, block_type, layers, num_classes)
    return model


def BaseTest():
    return _basenet("BaseTest", base_block, [1, 1])


def BottleTest():
    return _basenet("BottleTest", bottleneck_block, [1, 1])


def BaseNet34():
    return _basenet("BaseNet34", base_block, [3, 4, 6, 3])


def BaseNet50():
    return _basenet("BaseNet50", bottleneck_block, [3, 4, 6, 3])
