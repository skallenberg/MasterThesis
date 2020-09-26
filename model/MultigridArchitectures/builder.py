from model.BaseNet.blocks import *

from .net import MNANet
from utils.config import Config


def _mna_net(name, block_type, layers, residual=False, progressive=False):
    config = Config().get_instance()

    if config["Setup"]["Data"] == "cifar100":
        num_classes = 100
    else:
        num_classes = 10
    return MNANet(
        name, block_type, layers, num_classes, residual=residual, progressive=progressive,
    )


def MAN_Test():
    return _mna_net("MAN_Test", base_block, [1, 1, 1, 1, 1], progressive=True, residual=False)


def MAN_Bottle_Test():
    return _mna_net("MAN_Bottle_Test", bottleneck_block, [2, 2, 2, 2, 2])


def MAN_Res_Test():
    return _mna_net("MAN_Res_Test", base_block, [2, 2, 2, 2, 2], residual=True)


def MAN_Res_Bottle_Test():
    return _mna_net("MAN_Res_Bottle_Test", bottleneck_block, [2, 2, 2, 2, 2], residual=True)


def MG16():
    return _mna_net("MG16", base_block, [3, 4, 4, 4, 4],)


def R_MG16():
    return _mna_net("R_MG16", base_block, [3, 4, 4, 4, 4], progressive=False, residual=True)


def PMG16():
    return _mna_net("PMG16", base_block, [2, 3, 3, 3, 3], progressive=True, residual=False)


def R_PMG16():
    return _mna_net("R_PMG16", base_block, [1, 2, 2, 2, 1], progressive=True, residual=True)
