import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.config import Config

config = Config.get_instance()

data_name = config["Setup"]["Data"]


def initial_conv(channels_in):
    if data_name == "mnist":
        input_dim = 1
    else:
        input_dim = 3
    return nn.Conv2d(input_dim, channels_in, kernel_size=7, stride=2, padding=3, bias=False)


def conv_3x3(channels_in, channels_out, stride=1, groups=1, dilation=1):
    return nn.Conv2d(
        in_channels=channels_in,
        out_channels=channels_out,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        dilation=dilation,
        groups=groups,
        bias=False,
    )


def conv_1x1(channels_in, channels_out, stride=1):
    return nn.Conv2d(
        in_channels=channels_in,
        out_channels=channels_out,
        kernel_size=1,
        stride=stride,
        bias=False,
    )
