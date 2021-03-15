from .blocks import *
from .net import ResNet
from utils.config import Config


def _resnet(name, block_type, layers, **kwargs):
    config = Config().get_instance()

    if config["Setup"]["Data"] == "cifar100":
        num_classes = 100
    else:
        num_classes = 10
    model = ResNet(name, block_type, layers, num_classes, **kwargs)
    return model


def ResBaseTest():
    return _resnet("ResBaseTest", residual_base_block, [1, 1, 1, 1], sd="prog")


def ResBottleTest():
    return _resnet("ResBottleTest", residual_bottleneck_block, [1, 1])

def ResNet18(**kwargs):
    return _resnet("ResNet18", residual_base_block, [2, 2, 2, 2], **kwargs)

def ResNet34(**kwargs):
    return _resnet("ResNet34", residual_base_block, [3, 4, 6, 3], **kwargs)

def ResNet18_SD(**kwargs):
    kwargs["sd"] = "prog"
    return _resnet("ResNet18_SD", residual_base_block, [2, 2, 2, 2], **kwargs)

def ResNet34_SD(**kwargs):
    kwargs["sd"] = "prog"
    return _resnet("ResNet34_SD", residual_base_block, [3, 4, 6, 3], **kwargs)


def ResNet50():
    return _resnet("ResNet50", residual_bottleneck_block, [3, 4, 6, 3],sd="prog")
