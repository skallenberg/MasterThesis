import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from model.common import *

from .blocks import *
from .utils import interpolate

from utils.config import Config

config = Config.get_instance()

data_name = config["Setup"]["Data"]
alpha = config["Misc"]["CELU_alpha"]


class MGNet(nn.Module):
    def __init__(self, name, layers, num_classes, smoothing_steps=3):
        super().__init__()

        self.name = name
        self.writer = ""
        self.layers = layers
        self.channels_in = 64
        self.smoothing_steps = smoothing_steps
        if data_name == "mnist":
            self.conv0 = whitening_block(1, self.channels_in)
        else:
            self.conv0 = whitening_block(3, self.channels_in)

        self.bn0 = GhostBatchNorm(self.channels_in, config["DataLoader"]["BatchSize"] // 32)
        self.activation0 = nn.CELU(alpha)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.global_maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Linear(64 * (2 ** layers), num_classes)

        self.scale = config["Misc"]["FC_Scale"]

        self.mappings = self._set_data_feature_mapper()
        self.extractors = self._set_feature_extractors()
        self.feature_interpolations = self._set_interpolations()
        self.data_interpolations = self._set_interpolations()

        self._init_modules()

    def _init_modules(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _set_data_feature_mapper(self):
        mappings = nn.ModuleList()
        ch_in = self.channels_in
        for i in range(self.layers + 1):
            mappings.append(mapping_block(channels_in=ch_in))
            ch_in = ch_in * 2
        return mappings

    def _set_interpolations(self):
        interpolations = nn.ModuleList()
        ch_in = self.channels_in
        for i in range(self.layers):
            interpolations.append(
                nn.Conv2d(ch_in, ch_in * 2, kernel_size=3, padding=1, stride=2, groups=ch_in,)
            )
            ch_in *= 2
        return interpolations

    def _set_feature_extractors(self):
        extractors = nn.ModuleList()
        ch_in = self.channels_in
        for i in range(self.layers):
            grid_extractors = nn.ModuleList()
            for j in range(self.smoothing_steps):
                grid_extractors.append(extractor_block(channels_in=ch_in))
            extractors.append(grid_extractors)
            ch_in *= 2
        return extractors

    def _forward_impl(self, x):
        out = self.conv0(x)
        out = self.bn0(out)
        out = self.activation0(out)
        out = self.maxpool(out)
        f_ = out
        out_ = out * 0
        for i in range(self.layers):
            for j in range(self.smoothing_steps):
                out_ = out_ + self.extractors[i][j](f_ - self.mappings[i](out_))
            n_out_ = self.feature_interpolations[i](out_)
            f_ = self.data_interpolations[i](f_ - self.mappings[i](out_)) + self.mappings[i + 1](
                n_out_
            )
            out_ = n_out_

        out = f_
        out = self.global_maxpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        out *= self.scale

        return out

    def forward(self, x):
        out = self._forward_impl(x)
        return out


class VMGNet(MGNet):
    def __init__(self, name, layers, num_classes, smoothing_steps=3):
        super().__init__(name, layers, num_classes, smoothing_steps=3)

        self.prolongations = self._set_prolongation()
        self.extractors_2 = self._set_feature_extractors()

    def _set_prolongation(self):
        prolongations = nn.ModuleList()
        ch_in = self.channels_in
        for i in range(1, self.layers + 1):
            prolongations.append(
                interpolate(self.channels_in * 2 ** i, channel_scale=0.5, scale=2)
            )
        return prolongations

    def _forward_impl(self, x):
        out = self.conv0(x)
        out = self.bn0(out)
        out = self.activation0(out)
        out = self.maxpool(out)
        f_ = out
        out_ = out * 0
        out_list = []
        n_out_list = []
        f_list = [f_]

        for i in range(self.layers):
            for j in range(self.smoothing_steps):
                out_ = out_ + self.extractors[i][j](f_ - self.mappings[i](out_))
            out_list.append(out_)
            n_out_ = self.feature_interpolations[i](out_)
            n_out_list.append(n_out_)
            f_ = self.data_interpolations[i](f_ - self.mappings[i](out_)) + self.mappings[i + 1](
                n_out_
            )
            f_list.append(f_)
            out_ = n_out_

        for i in reversed(range(self.layers)):
            out_ = out_list[i] + self.prolongations[i](out_ - n_out_list[i])
            for j in range(self.smoothing_steps):
                out_ = out_ + self.extractors_2[i][j](f_list[i] - self.mappings[i](out_))

        out = out_
        out = self.global_maxpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        out *= self.scale

        return out

    def forward(self, x):
        out = self._forward_impl(x)
        return out


class FASMGNet(MGNet):
    def __init__(self, name, layers, num_classes, smoothing_steps=3, mode=1):
        super().__init__(name, layers, num_classes, smoothing_steps=3)
        self.mode = mode
        if mode == 1 or mode == 2:
            self.n_extractor_steps = 2 * layers - 1
        else:
            self.n_extractor_steps = 2 * layers + 1

        self.prolongations = self._set_prolongation()
        self.extractors_down = nn.ModuleList([self.extractors])
        self.extractors_up = nn.ModuleList()
        for i in range(self.n_extractor_steps // 2):
            self.extractors_down.append(self._set_feature_extractors())
            self.extractors_up.append(self._set_feature_extractors())

    def _set_prolongation(self):
        prolongations = nn.ModuleList()
        ch_in = self.channels_in
        for i in range(1, self.layers + 1):
            prolongations.append(
                interpolate(self.channels_in * 2 ** i, channel_scale=0.5, scale=2)
            )
        return prolongations

    def _mode_1_cycle(self, out):
        f_ = out
        out_ = out * 0
        out_list = []
        n_out_list = []
        f_list = [f_]

        for k in range(self.layers):
            for i in range(k, self.layers):
                for j in range(self.smoothing_steps):
                    out_ = out_ + self.extractors_down[k][i][j](f_ - self.mappings[i](out_))
                out_list.append(out_)
                n_out_ = self.feature_interpolations[i](out_)
                n_out_list.append(n_out_)
                f_ = self.data_interpolations[i](f_ - self.mappings[i](out_)) + self.mappings[
                    i + 1
                ](n_out_)
                f_list.append(f_)
                out_ = n_out_

            if len(n_out_list) < self.layers:
                addendum = [0] * (self.layers - len(n_out_list))
                n_out_list = [*addendum, *n_out_list]
                out_list = [*addendum, *out_list]
                f_list = [*addendum, *f_list]

            if k + 1 < self.layers:
                for i in reversed(range(k + 1, self.layers)):
                    out_part = self.prolongations[i](out_ - n_out_list[i])
                    out_ = out_list[i] + out_part
                    for j in range(self.smoothing_steps):
                        out_ = out_ + self.extractors_up[k][i][j](
                            f_list[i] - self.mappings[i](out_)
                        )

                f_ = f_list[k + 1]
                out_list = []
                n_out_list = []
                f_list = [f_]
        return out_, f_

    def _mode_2_cycle(self, out):
        f_ = out
        out_ = out * 0
        out_list = []
        n_out_list = []
        f_list = [f_]

        for k in reversed(range(self.layers)):
            for i in range(self.layers - k):
                for j in range(self.smoothing_steps):
                    out_ = out_ + self.extractors_down[k][i][j](f_ - self.mappings[i](out_))
                out_list.append(out_)
                n_out_ = self.feature_interpolations[i](out_)
                n_out_list.append(n_out_)
                f_ = self.data_interpolations[i](f_ - self.mappings[i](out_)) + self.mappings[
                    i + 1
                ](n_out_)
                f_list.append(f_)
                out_ = n_out_

            if len(n_out_list) < self.layers:
                addendum = [0] * (self.layers - len(n_out_list))
                n_out_list = [*addendum, *n_out_list]
                out_list = [*addendum, *out_list]
                f_list = [*addendum, *f_list]

            if k > 0:
                for i in reversed(range(0, self.layers - k)):
                    print(out_list[i].size())
                    print(self.prolongations[i](out_ - n_out_list[i]).size())
                    out_ = out_list[i] + self.prolongations[i](out_ - n_out_list[i])
                    for j in range(self.smoothing_steps):
                        out_ = out_ + self.extractors_up[len(self.extractors_up) - k][i][j](
                            f_list[i] - self.mappings[i](out_)
                        )

                f_ = f_list[0]
                out_list = []
                n_out_list = []
                f_list = [f_]
        return out_, f_

    def _cycle(self, out):
        if self.mode == 1:
            return self._mode_1_cycle(out)
        elif self.mode == 2:
            return self._mode_2_cycle(out)

    def _forward_impl(self, x):
        out = self.conv0(x)
        out = self.bn0(out)
        out = self.activation0(out)
        out = self.maxpool(out)

        out_, f_ = self._cycle(out)
        """# regular down path from fine 16x16 grid to coarse 2x2
        for i in range(self.layers):
            for j in range(self.smoothing_steps):
                out_ = out_ + self.extractors[i][j](f_ - self.mappings[i](out_))
            out_list.append(out_)
            n_out_ = self.feature_interpolations[i](out_)
            n_out_list.append(n_out_)
            f_ = self.data_interpolations[i](f_ - self.mappings[i](out_)) + self.mappings[i + 1](
                n_out_
            )
            f_list.append(f_)
            out_ = n_out_

        # first upsampling to 8x8
        for i in reversed(range(1, self.layers)):
            out_ = out_list[i] + self.prolongations[i](out_ - n_out_list[i])
            for j in range(self.smoothing_steps):
                out_ = out_ + self.extractors_2[i][j](f_list[i] - self.mappings[i](out_))

        f_ = f_list[1]

        # down to 2x2
        for i in range(1, self.layers):
            for j in range(self.smoothing_steps):
                out_ = out_ + self.extractors_3[i][j](f_ - self.mappings[i](out_))
            n_out_ = self.feature_interpolations[i](out_)
            f_ = self.data_interpolations[i](f_ - self.mappings[i](out_)) + self.mappings[i + 1](
                n_out_
            )
            out_ = n_out_

        # up to 4x4
        for i in reversed(range(2, self.layers)):
            out_ = out_list[i] + self.prolongations[i](out_ - n_out_list[i])
            for j in range(self.smoothing_steps):
                out_ = out_ + self.extractors_4[i][j](f_list[i] - self.mappings[i](out_))

        f_ = f_list[2]

        # down to 2x2
        for i in range(2, self.layers):
            for j in range(self.smoothing_steps):
                out_ = out_ + self.extractors_5[i][j](f_ - self.mappings[i](out_))
            n_out_ = self.feature_interpolations[i](out_)
            f_ = self.data_interpolations[i](f_ - self.mappings[i](out_)) + self.mappings[i + 1](
                n_out_
            )
            out_ = n_out_"""

        # return
        out = out_
        out = self.global_maxpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        out *= self.scale

        return out

    def forward(self, x):
        out = self._forward_impl(x)
        return out

