from .blocks import *
import torch
import torch.nn as nn
import torch.functional as F


class Inception_v4(nn.Module):
    def __init__(self, name, layers, num_classes=10):
        super().__init__()

        self.name = name
        self.writer = ""

        self.add_module("Stem", stem_v2())
        for i in range(layers[0]):
            self.add_module("InceptionA_%i" % (i), InceptionA(channels_in=384))

        self.add_module(
            "ReductionA", ReductionA(channels_in=384, k=192, l=224, m=256, n=384)
        )

        for i in range(layers[1]):
            self.add_module("InceptionB_%i" % (i), InceptionB(channels_in=1024))

        self.add_module("ReductionB", ReductionB(channels_in=1024))

        for i in range(layers[2]):
            self.add_module("InceptionC_%i" % (i), InceptionC(channels_in=1536))
        self.add_module("AvgPool", nn.AvgPool2d())

        self.add_module("Dropout", nn.Dropout2d(0.8))

        self.add_module("Linear", nn.Linear(1535, num_classes))

    def _forward_impl(self, x):
        out = x

        for m in self.modules():
            out = m(out)

        return out

    def forward(self, x):
        out = self._forward_impl(x)
        return out

