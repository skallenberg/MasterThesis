from model.BaseNetv2.blocks import *

from .net import FractalNetv2


def _fractal_netv2(
    name, block_type, layers, fractal_expansion=4, num_classes=10, drop_path=False, depthwise=False
):
    model = FractalNetv2(
        name,
        block_type,
        layers,
        fractal_expansion=fractal_expansion,
        num_classes=num_classes,
        drop_path=drop_path,
        depthwise=depthwise,
    )
    return model


def FractalTestv2():
    return _fractal_netv2("FractalTestv2", base_block, 2, drop_path=True)


def FractalNet3():
    return _fractal_net("FractalNet3", base_block, 3, drop_path=True)


def FractalNet4():
    return _fractal_net("FractalNet4", base_block, 4, drop_path=True)


def FractalNet5():
    return _fractal_net("FractalNet5", base_block, 5, drop_path=True)
