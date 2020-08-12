from .net import FractalNet
from model.BaseNet.blocks import *


def _fractal_net(
    name, block_type, layers, fractal_expansion=4, num_classes=10, drop_path=False
):
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
    return _fractal_net("FractalTest", bottleneck_block, 2, drop_path=True)


def FractalNet3():
    return _fractal_net("FractalNet3", base_block, 3)


def FractalNet4():
    return _fractal_net("FractalNet4", base_block, 4)


def FractalNet5():
    return _fractal_net("FractalNet5", base_block, 5)
