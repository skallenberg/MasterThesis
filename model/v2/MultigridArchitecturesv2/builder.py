from model.v2.BaseNetv2.blocks import *

from .net import MNANetv2


def _mna_netv2(name, block_type, layers, num_classes=10, residual=False, progressive=False):
    return MNANetv2(
        name,
        block_type,
        layers,
        num_classes,
        residual=residual,
        progressive=progressive,
        depthwise=False,
    )


def MAN_Testv2():
    return _mna_netv2("MAN_Test", base_block, [1, 1, 1, 1, 1], progressive=True, residual=False)


def MAN_Bottle_Testv2():
    return _mna_netv2("MAN_Bottle_Test", bottleneck_block, [2, 2, 2, 2, 2])


def MAN_Res_Testv2():
    return _mna_netv2("MAN_Res_Test", base_block, [2, 2, 2, 2, 2], residual=True)


def MAN_Res_Bottle_Testv2():
    return _mna_netv2("MAN_Res_Bottle_Test", bottleneck_block, [2, 2, 2, 2, 2], residual=True)


def MG16v2():
    return _mna_netv2("MG16", base_block, [3, 4, 4, 4, 4],)


def R_MG16v2():
    return _mna_netv2("R_MG16", base_block, [3, 4, 4, 4, 4], progressive=False, residual=True)


def PMG16v2():
    return _mna_netv2("PMG16", base_block, [2, 3, 3, 3, 3], progressive=True, residual=False)


def R_PMG16v2():
    return _mna_netv2("R_PMG16", base_block, [1, 2, 2, 2, 1], progressive=True, residual=True)
