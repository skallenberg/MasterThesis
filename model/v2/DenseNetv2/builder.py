from .blocks import *
from .net import DenseNetv2


def _densenetv2(name, block_type, layers, num_classes=10):
    model = DenseNetv2(name, block_type, layers, num_classes=num_classes)
    return model


def DenseTestv2():
    return _densenetv2("DenseTestv2", dense_base_unit, [2, 2])


def DenseNet34v2():
    return _densenetv2("DenseNet34v2", dense_base_unit, [3, 4, 6, 3])


def DenseNet50v2():
    return _densenetv2("DenseNet50", dense_alternative_bottleneck_unit, [3, 4, 6, 3])


def DenseNet50_2v2():
    return _densenetv2("DenseNet50_2", dense_orig_bottleneck_unit, [3, 4, 6, 3])


def DenseNet121v2():
    return _densenetv2("DenseNet121v2", dense_orig_bottleneck_unit, [6, 12, 12, 16])


def DenseNet161v2():
    return _densenetv2("DenseNet161v2", dense_orig_bottleneck_unit, [6, 12, 36, 24])


def DenseNet169v2():
    return _densenetv2("DenseNet169v2", dense_orig_bottleneck_unit, [6, 12, 32, 32])


def DenseNet201v2():
    return _densenetv2("DenseNet201v2", dense_orig_bottleneck_unit, [6, 12, 48, 32])
