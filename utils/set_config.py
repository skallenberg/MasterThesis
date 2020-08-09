import torch.optim as optim
from utils.config import Config
from model import resnet, basenet, densenet, fractalnet, NiN, RoR, MANet

config = Config.get_instance()

opt = config["Optimizer"]["Type"]
arch = config["Setup"]["Architecture"]


def choose_optimizer(net):
    if opt == "SDG":
        return optim.SGD(
            net.parameters(),
            lr=config["Optimizer"]["LearnRate"],
            momentum=config["Optimizer"]["Momentum"],
            nesterov=config["Optimizer"]["SDGNesterov"],
            weight_decay=config["Optimizer"]["WeightDecay"],
        )
    if opt == "Adam":
        return optim.Adam(
            net.parameters(),
            lr=config["Optimizer"]["LearnRate"],
            weight_decay=config["Optimizer"]["WeightDecay"],
            amsgrad=config["Optimizer"]["AdamAMSGrad"],
        )
    if opt == "AdaGrad":
        return optim.Adagrad(
            net.parameters(),
            lr=config["Optimizer"]["LearnRate"],
            lr_decay=config["Optimizer"]["AdaGradLRDecay"],
            weight_decay=config["Optimizer"]["WeightDecay"],
        )


def choose_architecture():
    if arch == "BaseTest":
        return basenet.BaseTest()

    if arch == "BottleTest":
        return basenet.BottleTest()

    if arch == "ResBaseTest":
        return resnet.ResBaseTest()

    if arch == "ResBottleTest":
        return resnet.ResBottleTest()

    if arch == "NeXtTest":
        return resnet.NeXtTest()

    if arch == "WideTest":
        return resnet.WideTest()

    if arch == "BaseNet34":
        return basenet.BaseNet34()

    if arch == "BaseNet50":
        return basenet.BaseNet50()

    if arch == "ResNet34":
        return resnet.ResNet34()

    if arch == "ResNet50":
        return resnet.ResNet50()

    if arch == "ResNeXt50":
        return resnet.ResNeXt50()

    if arch == "WideResNet50":
        return resnet.WideResNet50()

    if arch == "DenseTest":
        return densenet.DenseTest()

    if arch == "DenseNet34":
        return densenet.DenseNet34()

    if arch == "DenseNet50":
        return densenet.DenseNet50()

    if arch == "DenseNet50_2":
        return densenet.DenseNet50_2()

    if arch == "DenseNet121":
        return densenet.DenseNet121()

    if arch == "DenseNet161":
        return densenet.DenseNet161()

    if arch == "DenseNet169":
        return densenet.DenseNet169()

    if arch == "DenseNet201":
        return densenet.DenseNet201()

    if arch == "FractalTest":
        return fractalnet.FractalTest()

    if arch == "FractalNet3":
        return fractalnet.FractalNet3()

    if arch == "FractalNet4":
        return fractalnet.FractalNet4()

    if arch == "FractalNet5":
        return fractalnet.FractalNet5()

    if arch == "NiN_BaseTest":
        return NiN.NiN_BaseTest()

    if arch == "NiN_BottleTest":
        return NiN.NiN_BottleTest()

    if arch == "NiN_Net34":
        return NiN.NiN_Net34()

    if arch == "NiN_Net50":
        return NiN.NiN_Net50()

    if arch == "RoR_BaseTest":
        return RoR.RoR_BaseTest()

    if arch == "RoR_BottleTest":
        return RoR.RoR_BottleTest()

    if arch == "RoR_Net34":
        return RoR.RoR_Net34()

    if arch == "RoR_Net50":
        return RoR.RoR_Net50()

    if arch == "RoR_NeXt50":
        return RoR.RoR_NeXt50()

    if arch == "RoR_WideResNet50":
        return RoR.RoR_WideResNet50()

    if arch == "MAN_Test":
        return MANet.MAN_Test()

    if arch == "MAN_Bottle_Test":
        return MANet.MAN_Bottle_Test()

    if arch == "MAN_Res_Test":
        return MANet.MAN_Res_Test()

    if arch == "MAN_Res_Bottle_Test":
        return MANet.MAN_Res_Bottle_Test()

    if arch == "StochasticDepthTest":
        return resnet.StochasticDepthTest()

    if arch == "MG16":
        return MANet.MG16()

    if arch == "R_MG16":
        return MANet.R_MG16()

    if arch == "PMG16":
        return MANet.PMG16()

    if arch == "R_PMG16":
        return MANet.R_PMG16()

    else:
        raise ValueError("Model not yet implemented")

