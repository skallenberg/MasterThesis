from model.BaseNet.blocks import *

from .net import FractalNet
from utils.config import Config


def _fractal_net(
    name, block_type, layers, fractal_expansion=4, drop_path=False,
):
    config = Config().get_instance()

    if config["Setup"]["Data"] == "cifar100":
        num_classes = 100
    else:
        num_classes = 10
    model = FractalNet(
        name,
        block_type,
        layers,
        fractal_expansion=fractal_expansion,
        num_classes=num_classes,
        drop_path=drop_path,
    )
    return model


def FractalTest():
    return _fractal_net("FractalTest", base_block, 2, drop_path=True)


def FractalNet3():
    return _fractal_net("FractalNet3", base_block, 3, drop_path=True)


def FractalNet4():
    return _fractal_net("FractalNet4", base_block, 4, drop_path=True)


def FractalNet5():
    return _fractal_net("FractalNet5", base_block, 5, drop_path=True)
