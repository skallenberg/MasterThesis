from model.BaseNet.blocks import *

from .net import *
from utils.config import Config


def _MGNet(name, layers, **kwargs):
    config = Config().get_instance()

    if config["Setup"]["Data"] == "cifar100":
        num_classes = 100
    else:
        num_classes = 10
    model = MGNet(name, layers, num_classes, **kwargs)
    return model


def _FASMGNet(name, layers, **kwargs):
    config = Config().get_instance()

    if config["Setup"]["Data"] == "cifar100":
        num_classes = 100
    else:
        num_classes = 10
    model = FASMGNet(name, layers, num_classes, **kwargs)
    return model


def _VMGNet(name, layers, **kwargs):
    config = Config().get_instance()

    if config["Setup"]["Data"] == "cifar100":
        num_classes = 100
    else:
        num_classes = 10
    model = VMGNet(name, layers, num_classes, **kwargs)
    return model


def MGNetTest(**kwargs):
    return _MGNet("MGNetTest", 3, **kwargs)


def FASMGNetTest1(**kwargs):
    kwargs["mode"] = 1
    kwargs["smoothing_steps"] = 2
    kwargs["batch_norm"] = True
    return _FASMGNet("FASMGNetTest1", 3, **kwargs)

def FASMGNetTwoGrid_1(**kwargs):
    kwargs["mode"] = 1
    kwargs["smoothing_steps"] = 2
    kwargs["batch_norm"] = False
    return _FASMGNet("FASMGNetTwoGrid_1", 2, **kwargs)

def FASMGNet9_1(**kwargs):
    kwargs["mode"] = 1
    kwargs["smoothing_steps"] = 2
    kwargs["batch_norm"] = False
    return _FASMGNet("FASMGNet9_1", 3, **kwargs)

def FASMGNet12_1(**kwargs):
    kwargs["mode"] = 1
    kwargs["smoothing_steps"] = 2
    kwargs["batch_norm"] = False
    return _FASMGNet("FASMGNet12_1", 4, **kwargs)

def FASMGNet18_1(**kwargs):
    kwargs["mode"] = 1
    kwargs["smoothing_steps"] = 2
    kwargs["batch_norm"] = False
    return _FASMGNet("FASMGNet18_1", 5, **kwargs)

def FASMGNet34_1(**kwargs):
    kwargs["mode"] = 1
    kwargs["smoothing_steps"] = 4
    kwargs["batch_norm"] = False
    return _FASMGNet("FASMGNet34_1", 5, **kwargs)

def FASMGNetTwoGrid_2(**kwargs):
    kwargs["mode"] = 2
    kwargs["smoothing_steps"] = 2
    kwargs["batch_norm"] = False
    return _FASMGNet("FASMGNetTwoGrid_2", 2, **kwargs)

def FASMGNet9_2(**kwargs):
    kwargs["mode"] = 2
    kwargs["smoothing_steps"] = 2
    kwargs["batch_norm"] = False
    return _FASMGNet("FASMGNet9_2", 3, **kwargs)

def FASMGNet12_2(**kwargs):
    kwargs["mode"] = 1
    kwargs["smoothing_steps"] = 2
    kwargs["batch_norm"] = False
    return _FASMGNet("FASMGNet12_2", 4, **kwargs)

def FASMGNet18_2(**kwargs):
    kwargs["mode"] = 2
    kwargs["smoothing_steps"] = 2
    kwargs["batch_norm"] = False
    return _FASMGNet("FASMGNet18_2", 5, **kwargs)

def FASMGNet34_2(**kwargs):
    kwargs["mode"] = 2
    kwargs["smoothing_steps"] = 4
    kwargs["batch_norm"] = False
    return _FASMGNet("FASMGNet34_2", 5, **kwargs)

def FASMGNetTest3(**kwargs):
    kwargs["mode"] = 3
    kwargs["smoothing_steps"] = 2
    return _FASMGNet("FASMGNetTest3", 3, **kwargs)


def VMGNetTest(**kwargs):
    kwargs["smoothing_steps"] = 2
    return _VMGNet("VMGNetTest", 3, **kwargs)

def MGNetTwoGrid(**kwargs):
    kwargs["smoothing_steps"] = 2
    return _MGNet("MGNetTwoGrid", 2, **kwargs)

def MGNet9(**kwargs):
    kwargs["smoothing_steps"] = 2
    return _MGNet("MGNet9", 3, **kwargs)

def MGNet12(**kwargs):
    kwargs["smoothing_steps"] = 2
    return _MGNet("MGNet12", 4, **kwargs)

def MGNet18(**kwargs):
    kwargs["smoothing_steps"] = 2
    return _MGNet("MGNet18", 5, **kwargs)


def MGNet34(**kwargs):
    kwargs["smoothing_steps"] = 4
    return _MGNet("MGNet34", 5, **kwargs)
