from model.v2.BaseNetv2.blocks import *

from .net import *
from utils.config import Config

config = Config.get_instance()

if config["Setup"]["Data"] == "cifar100":
    num_classes = 100
else:
    num_classes = 10


def _MGNet(name, layers, num_classes=num_classes, **kwargs):
    model = MGNet(name, layers, num_classes, **kwargs)
    return model


def _FASMGNet(name, layers, num_classes=num_classes, **kwargs):
    model = FASMGNet(name, layers, num_classes, **kwargs)
    return model


def MGNetTest(**kwargs):
    return _MGNet("MGNetTest", 3, **kwargs)


def FASMGNetTest(**kwargs):
    kwargs["mode"] = 1
    kwargs["smoothing_steps"] = 2
    return _FASMGNet("VMGNetTest", 3, **kwargs)


def MGNet34(**kwargs):
    kwargs["smoothing_steps"] = 4
    return _MGNet("MGNetNet34", 4, **kwargs)


def MGNet50(**kwargs):
    kwargs["smoothing_steps"] = 4
    return _MGNet("MGNetTest", 5, **kwargs)
