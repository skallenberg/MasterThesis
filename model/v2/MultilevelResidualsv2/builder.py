from model.v2.ResNetv2.blocks import *

from .net import MRNNetv2


def _mrn_netv2(name, block_type, layers, num_classes=10, **kwargs):
    model = MRNNetv2(name, block_type, layers, num_classes, **kwargs)
    return model


def MRN_BaseTestv2():
    return _mrn_netv2("MRN_BaseTestv2", residual_base_block, [1, 1])


def MRN_BottleTestv2():
    return _mrn_netv2("MRN_BottleTestv2", residual_bottleneck_block, [1, 1])


def MRN_Net34v2():
    return _mrn_netv2("MRN_Net34v2", residual_base_block, [3, 4, 6, 3])


def MRN_Net50v2():
    return _mrn_netv2("MRN_Net50v2", residual_bottleneck_block, [3, 4, 6, 3])


def MRN_NeXt50v2(**kwargs):
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 4
    return _mrn_netv2("MRN_NeXt50v2", residual_bottleneck_block, [3, 4, 6, 3], **kwargs)


def MRN_WideResNet50v2(**kwargs):
    kwargs["width_per_group"] = 64 * 2
    return _mrn_netv2("MRN_WideResNet50v2", residual_bottleneck_block, [3, 4, 6, 3], **kwargs)
