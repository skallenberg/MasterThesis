from .blocks import *
from .net import BaseNet
from utils.config import Config


def _basenet(name, block_type, layers, **kwargs):
    config = Config().get_instance()

    if config["Setup"]["Data"] == "cifar100":
        num_classes = 100
    else:
        num_classes = 10
    model = BaseNet(
        name=name, block_type=block_type, layers=layers, num_classes=num_classes, **kwargs
    )
    return model


def BaseTest(**kwargs):
    return _basenet("BaseTest", base_block, [1, 2, 1, 1], **kwargs)


def BottleTest(**kwargs):
    return _basenet("BottleTest", bottleneck_block, [1, 1], **kwargs)


def BaseNet34(**kwargs):
    return _basenet("BaseNet34", base_block, [3, 4, 6, 3])


def BaseNet50(**kwargs):
    return _basenet("BaseNet50", bottleneck_block, [3, 4, 6, 3])
