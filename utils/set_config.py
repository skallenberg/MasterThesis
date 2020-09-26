import torch.optim as optim

from model import *

from utils.config import Config

config = Config.get_instance()

opt = config["Optimizer"]["Type"]
arch = config["Setup"]["Architecture"]


def choose_optimizer(net):
    if opt == "SGD":
        return optim.SGD(
            net.parameters(),
            lr=config["Optimizer"]["LearnRate"],
            momentum=config["Optimizer"]["Momentum"],
            nesterov=config["Optimizer"]["SGDNesterov"],
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

    if opt == "AdamW":
        return optim.AdamW(
            net.parameters(),
            lr=config["Optimizer"]["LearnRate"],
            weight_decay=config["Optimizer"]["WeightDecay"],
            amsgrad=config["Optimizer"]["AdamAMSGrad"],
        )


def choose_architecture():

    try:
        return eval(arch + "()")
    except:
        raise ValueError("Model not yet implemented")
