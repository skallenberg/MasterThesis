from .blocks import *
from .net import NiN_Net
from utils.config import Config


def _NiN_Net(name, block_type, layers):
    config = Config().get_instance()

    if config["Setup"]["Data"] == "cifar100":
        num_classes = 100
    else:
        num_classes = 10
    model = NiN_Net(name, block_type, layers, num_classes)
    return model


def NiN_BaseTest():
    return _NiN_Net("NiN_BaseTest", NiN_base_block, [1, 1])


def NiN_BottleTest():
    return _NiN_Net("NiN_BottleTest", NiN_bottleneck_block, [1, 1])


def NiN_Net34():
    return _NiN_Net("NiN_Net34", NiN_base_block, [3, 4, 6, 3])


def NiN_Net50():
    return _NiN_Net("NiN_Net50", NiN_bottleneck_block, [3, 4, 6, 3])
