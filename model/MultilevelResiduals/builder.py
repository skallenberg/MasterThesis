from model.ResNet.blocks import *

from .net import MRNNet
from utils.config import Config


def _mrn_net(name, block_type, layers, **kwargs):
    config = Config().get_instance()

    if config["Setup"]["Data"] == "cifar100":
        num_classes = 100
    else:
        num_classes = 10
    model = MRNNet(name, block_type, layers, num_classes, **kwargs)
    return model


def MRN_BaseTest():
    return _mrn_net("MRN_BaseTest", residual_base_block, [1, 1, 1, 1])


def MRN_BottleTest():
    return _mrn_net("MRN_BottleTest", residual_bottleneck_block, [1, 1, 1, 1])


def MRN_Net34():
    return _mrn_net("MRN_Net34", residual_base_block, [3, 4, 6, 3])


def MRN_Net50():
    return _mrn_net("MRN_Net50", residual_bottleneck_block, [3, 4, 6, 3])
