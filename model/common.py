import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.config import Config
from utils.eigenval import compute_eigenvalues
import os


class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, weight=True, bias=True):
        super().__init__(num_features, eps=eps, momentum=momentum)
        self.weight.data.fill_(1.0)
        self.bias.data.fill_(0.0)
        self.weight.requires_grad = weight
        self.bias.requires_grad = bias


class GhostBatchNorm(BatchNorm):
    def __init__(self, num_features, num_splits, **kw):
        super().__init__(num_features, **kw)
        self.num_splits = num_splits
        self.register_buffer("running_mean", torch.zeros(num_features * self.num_splits))
        self.register_buffer("running_var", torch.ones(num_features * self.num_splits))

    def train(self, mode=True):
        if (self.training is True) and (
            mode is False
        ):  # lazily collate stats when we are going to use them
            self.running_mean = torch.mean(
                self.running_mean.view(self.num_splits, self.num_features), dim=0
            ).repeat(self.num_splits)
            self.running_var = torch.mean(
                self.running_var.view(self.num_splits, self.num_features), dim=0
            ).repeat(self.num_splits)
        return super().train(mode)

    def forward(self, input):
        N, C, H, W = input.shape
        if self.training or not self.track_running_stats:
            return F.batch_norm(
                input.view(-1, C * self.num_splits, H, W),
                self.running_mean,
                self.running_var,
                self.weight.repeat(self.num_splits),
                self.bias.repeat(self.num_splits),
                True,
                self.momentum,
                self.eps,
            ).view(N, C, H, W)
        else:
            return F.batch_norm(
                input,
                self.running_mean[: self.num_features],
                self.running_var[: self.num_features],
                self.weight,
                self.bias,
                False,
                self.momentum,
                self.eps,
            )


def initial_conv(channels_out):
    config = Config().get_instance()

    if config["Setup"]["Data"] == "mnist":
        input_dim = 1
    else:
        input_dim = 3
    if config["Misc"]["Depthwise"]:
        groups = channels_out
    else:
        groups = 1
    return nn.Conv2d(
        input_dim,
        channels_out,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=config["Misc"]["ConvolutionBias"],
        groups=groups,
    )


def conv_3x3(channels_in, channels_out, stride=1, padding=1, groups=None):
    config = Config().get_instance()

    if groups is None:
        if config["Misc"]["Depthwise"] and channels_out % channels_in == 0:
            groups = channels_in
        else:
            groups = 1
    return nn.Conv2d(
        in_channels=channels_in,
        out_channels=channels_out,
        kernel_size=3,
        stride=stride,
        padding=padding,
        groups=groups,
        bias=config["Misc"]["ConvolutionBias"],
    )


def conv_1x1(channels_in, channels_out, stride=1):
    config = Config().get_instance()

    if config["Misc"]["Depthwise"] and channels_out % channels_in == 0:
        groups = channels_in
    else:
        groups = 1
    return nn.Conv2d(
        in_channels=channels_in,
        out_channels=channels_out,
        kernel_size=1,
        stride=stride,
        bias=config["Misc"]["ConvolutionBias"],
        groups=groups,
    )


def WhiteningBlock(c_out, eps=1e-2):
    config = Config().get_instance()

    base_path = "./data/datasets/eigenvals/" + config["Setup"]["Data"]
    eigen_path_1 = base_path + "_1.pt"
    eigen_path_2 = base_path + "_2.pt"
    if os.path.isfile(eigen_path_1) and os.path.isfile(eigen_path_2):
        V = torch.load(eigen_path_1)
        W = torch.load(eigen_path_2)
    else:
        V, W = compute_eigenvalues()

    if config["Setup"]["Data"] == "mnist":
        c_in = 1
    else:
        c_in = 3

    filt = nn.Conv2d(
        c_in, 27, kernel_size=(3, 3), padding=(1, 1), bias=config["Misc"]["ConvolutionBias"],
    )
    filt.weight.data = W / torch.sqrt(V + eps)[:, None, None, None]
    filt.weight.requires_grad = False

    if config["Misc"]["GhostBatchNorm"]:
        bn = GhostBatchNorm(c_out, config["DataLoader"]["BatchSize"] // 32)
    else:
        bn = nn.BatchNorm2d(c_out)
    if config["Misc"]["UseCELU"]:
        act = nn.CELU(config["Misc"]["CELU_alpha"])
    else:
        act = nn.ReLU(inplace=True)

    return nn.Sequential(
        filt,
        nn.Conv2d(27, c_out, kernel_size=(1, 1), bias=config["Misc"]["ConvolutionBias"]),
        bn,
        act,
    )


def init_block(channels_out):
    config = Config().get_instance()

    if config["Misc"]["WhiteningBlock"]:
        conv = WhiteningBlock(channels_out)
    else:
        conv = initial_conv(channels_out)
    if config["Misc"]["GhostBatchNorm"]:
        bn = GhostBatchNorm(channels_out, config["DataLoader"]["BatchSize"] // 32)
    else:
        bn = nn.BatchNorm2d(channels_out)
    if config["Misc"]["UseCELU"]:
        act = nn.CELU(config["Misc"]["CELU_alpha"])
    else:
        act = nn.ReLU(inplace=True)
    return nn.Sequential(*[conv, bn, act])


def conv_bn_act(channels_in, channels_out, stride=1, extra=None):
    config = Config().get_instance()

    if extra is None:
        extra = config["Misc"]["TwoConvsPerBlock"]

    if stride > 1:
        if config["Misc"]["StrideToPooling"]:
            conv = conv_3x3(channels_in, channels_out)
            mpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            conv = conv_3x3(channels_in, channels_out, stride=stride)
            mpool = nn.Identity()
    else:
        conv = conv_3x3(channels_in, channels_out)
        mpool = nn.Identity()

    if config["Misc"]["GhostBatchNorm"]:
        bn = GhostBatchNorm(channels_out, config["DataLoader"]["BatchSize"] // 32)
    else:
        bn = nn.BatchNorm2d(channels_out)

    if config["Misc"]["UseCELU"]:
        act = nn.CELU(config["Misc"]["CELU_alpha"])
    else:
        act = nn.ReLU(inplace=True)

    if extra:
        extra = conv_bn_act(channels_out, channels_out, extra=False)
    else:
        extra = nn.Identity()

    if config["Misc"]["PoolBeforeBN"]:
        return nn.Sequential(*[conv, mpool, bn, act, extra])
    else:
        return nn.Sequential(*[conv, bn, act, mpool, extra])


def conv_bn_act_bottleneck(channels_in, channels_out, stride=1):
    config = Config().get_instance()

    if stride > 1:
        if config["Misc"]["StrideToPooling"]:
            conv_1 = conv_1x1(channels_in, channels_out)
            mpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            conv_1 = conv_1x1(channels_in, channels_out, stride=stride)
            mpool = nn.Identity()
    else:
        conv_1 = conv_1x1(channels_in, channels_out)
        mpool = nn.Identity()

    conv_2 = conv_3x3(channels_out, channels_out)
    conv_3 = conv_1x1(channels_out, channels_out * 4)

    if config["Misc"]["GhostBatchNorm"]:
        bn_1 = GhostBatchNorm(channels_out, config["DataLoader"]["BatchSize"] // 32)
        bn_2 = GhostBatchNorm(channels_out, config["DataLoader"]["BatchSize"] // 32)
        bn_3 = GhostBatchNorm(channels_out * 4, config["DataLoader"]["BatchSize"] // 32)
    else:
        bn_1 = nn.BatchNorm2d(channels_out)
        bn_2 = nn.BatchNorm2d(channels_out)
        bn_3 = nn.BatchNorm2d(channels_out * 3)

    if config["Misc"]["UseCELU"]:
        act = nn.CELU(config["Misc"]["CELU_alpha"])
    else:
        act = nn.ReLU(inplace=True)

    if config["Misc"]["PoolBeforeBN"]:
        return nn.Sequential(*[conv_1, mpool, bn_1, act, conv_2, bn_2, act, conv_3, bn_3, act])
    else:
        return nn.Sequential(*[conv_1, bn_1, act, mpool, conv_2, bn_2, act, conv_3, bn_3, act])

