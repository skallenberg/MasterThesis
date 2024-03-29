from .blocks import *
from .net import DenseNet
from utils.config import Config


def _densenet(name, block_type, layers, **kwargs):
    config = Config().get_instance()

    if config["Setup"]["Data"] == "cifar100":
        num_classes = 100
    else:
        num_classes = 10
    model = DenseNet(name, block_type, layers, num_classes=num_classes, **kwargs)
    return model


def DenseTest():
    return _densenet("DenseTest", dense_base_unit, [2, 2])


def DenseNet34():
    return _densenet("DenseNet34", dense_base_unit, [3, 4, 6, 3])


def DenseNet50():
    return _densenet("DenseNet50", dense_alternative_bottleneck_unit, [3, 4, 6, 3])


def DenseNet50_2():
    return _densenet("DenseNet50_2", dense_orig_bottleneck_unit, [3, 4, 6, 3])


def DenseNet121():
    return _densenet("DenseNet121", dense_orig_bottleneck_unit, [6, 12, 24, 16])

def DenseNet121DropOut(**kwargs):
    kwargs["drop_rate"] = 0.5
    return _densenet("DenseNet121DropOut", dense_orig_bottleneck_unit, [6, 12, 24, 16], **kwargs)


def DenseNet161():
    return _densenet("DenseNet161", dense_orig_bottleneck_unit, [6, 12, 36, 24])


def DenseNet169():
    return _densenet("DenseNet169", dense_orig_bottleneck_unit, [6, 12, 32, 32])


def DenseNet201():
    return _densenet("DenseNet201", dense_orig_bottleneck_unit, [6, 12, 48, 32])
