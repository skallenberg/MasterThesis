import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.config import Config
from utils.eigenval import compute_eigenvalues
import os

config = Config.get_instance()

data_name = config["Setup"]["Data"]
alpha = config["Misc"]["CELU_alpha"]


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


def conv_1x1(channels_in, channels_out, stride=1, groups=1):
    return nn.Conv2d(
        in_channels=channels_in,
        out_channels=channels_out,
        kernel_size=1,
        stride=stride,
        bias=False,
        groups=groups,
    )


base_path = "./data/datasets/eigenvals/" + config["Setup"]["Data"]
eigen_path_1 = base_path + "_1.pt"
eigen_path_2 = base_path + "_2.pt"
if os.path.isfile(eigen_path_1) and os.path.isfile(eigen_path_2):
    V = torch.load(eigen_path_1)
    W = torch.load(eigen_path_2)
else:
    V, W = compute_eigenvalues()


def whitening_block(c_in, c_out, eps=1e-2):
    filt = nn.Conv2d(3, 27, kernel_size=(3, 3), padding=(1, 1), bias=False)
    filt.weight.data = W / torch.sqrt(V + eps)[:, None, None, None]
    filt.weight.requires_grad = False

    return nn.Sequential(
        filt,
        nn.Conv2d(27, c_out, kernel_size=(1, 1), bias=False),
        GhostBatchNorm(c_out, 2),
        nn.CELU(alpha),
    )
