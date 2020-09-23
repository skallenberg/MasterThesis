import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.extractors_3 = self._set_feature_extractors()
        self.extractors_4 = self._set_feature_extractors()
        self.extractors_5 = self._set_feature_extractors()

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

        for i in reversed(range(1, self.layers)):
            out_ = out_list[i] + self.prolongations[i](out_ - n_out_list[i])
            for j in range(self.smoothing_steps):
                out_ = out_ + self.extractors_2[i][j](f_list[i] - self.mappings[i](out_))

        f_ = f_list[1]

        for i in range(1, self.layers):
            for j in range(self.smoothing_steps):
                out_ = out_ + self.extractors_3[i][j](f_ - self.mappings[i](out_))
            n_out_ = self.feature_interpolations[i](out_)
            f_ = self.data_interpolations[i](f_ - self.mappings[i](out_)) + self.mappings[i + 1](
                n_out_
            )
            out_ = n_out_

        for i in reversed(range(2, self.layers)):
            out_ = out_list[i] + self.prolongations[i](out_ - n_out_list[i])
            for j in range(self.smoothing_steps):
                out_ = out_ + self.extractors_4[i][j](f_list[i] - self.mappings[i](out_))

        f_ = f_list[2]

        for i in range(2, self.layers):
            for j in range(self.smoothing_steps):
                out_ = out_ + self.extractors_5[i][j](f_ - self.mappings[i](out_))
            n_out_ = self.feature_interpolations[i](out_)
            f_ = self.data_interpolations[i](f_ - self.mappings[i](out_)) + self.mappings[i + 1](
                n_out_
            )
            out_ = n_out_

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
        self.n_extractors = self.layers * 2 - 1

        self.extractors_down, self.extractors_up = self._set_steps_down()

        self.prolongations = self._set_prolongation()

    def _set_prolongation(self):
        prolongations = nn.ModuleList()
        ch_in = self.channels_in
        for i in range(1, self.layers + 1):
            prolongations.append(
                interpolate(self.channels_in * 2 ** i, channel_scale=0.5, scale=2)
            )
        return prolongations

    def _set_steps_down(self):
        if self.mode == 1 or self.mode == 3:
            extractor_list_down = nn.ModuleList()
            extractor_list_up = nn.ModuleList()
            for i in range(self.n_extractors + 1):
                extractor_list_down.append(
                    self._set_stepwise_feature_extractors(
                        i, self.layers, self.channels_in * 2 ** i
                    )
                )
            for i in range(self.n_extractors + 1):
                extractor_list_up.append(
                    self._set_stepwise_feature_extractors(
                        0, self.layers - i, self.channels_in * 2 ** i
                    )
                )
            if self.mode == 3:
                extractor_list_down = reversed(extractor_list_down)
                extractor_list_up = reversed(extractor_list_up)
        if self.mode == 2:
            extractor_list_down = nn.ModuleList()
            extractor_list_up = nn.ModuleList()
            for i in range(self.n_extractors + 1):
                extractor_list_down.append(
                    self._set_stepwise_feature_extractors(
                        i, self.layers, self.channels_in * 2 ** i
                    )
                )
            for i in range(self.n_extractors + 1):
                extractor_list_up.append(
                    self._set_stepwise_feature_extractors(
                        i, self.layers, self.channels_in * 2 ** i
                    )
                )
            extractor_list_down = reversed(extractor_list_down)
            extractor_list_down = nn.ModuleList(
                *[
                    self._set_stepwise_feature_extractors(0, self.layers, self.channels_in),
                    *extractor_list_down,
                ]
            )
            extractor_list_up = reversed(extractor_list_up)
        return extractor_list_down, extractor_list_up

    def _set_stepwise_feature_extractors(self, start, stop, ch_in):
        extractors = nn.ModuleList()
        for i in range(start, stop):
            grid_extractors = nn.ModuleList()
            for j in range(self.smoothing_steps):
                grid_extractors.append(extractor_block(channels_in=ch_in))
            extractors.append(grid_extractors)
            ch_in *= 2
        return extractors

    def _cycle_down(self, extractors, start, stop, out_, f_):
        out_list = []
        n_out_list = []
        f_list = [f_]
        iterator = 0
        for i in range(start, stop):
            for j in range(self.smoothing_steps):
                out_part = f_ - self.mappings[i](out_)
                out_part2 = extractors[iterator][j](out_part)
                out_ = out_ + out_part2
            out_list.append(out_)
            n_out_ = self.feature_interpolations[i](out_)
            n_out_list.append(n_out_)
            f_ = self.data_interpolations[i](f_ - self.mappings[i](out_)) + self.mappings[i + 1](
                n_out_
            )
            f_list.append(f_)
            out_ = n_out_
            iterator += 1
        return out_, f_, out_list, n_out_list, f_list

    def _cycle_up(self, extractors, start, stop, out_, out_list, n_out_list, f_list):
        # print("CYCLING UP")
        # print("IN SIZE")
        # print(out_.size())
        iterator = len(out_list) - 1
        iterator_2 = len(extractors) - 1
        for i in reversed(range(start, stop)):
            print(i)
            print(len(out_list))
            out_ = out_list[iterator] + self.prolongations[i](out_ - n_out_list[iterator])
            iterator = iterator - 1
            # print("STEP SIZE")
            # print(out_.size())
            # print("STEP:", i)
            for j in range(self.smoothing_steps):
                out_ = out_ + extractors[iterator_2][j](f_list[i] - self.mappings[i](out_))
            iterator_2 = iterator_2 - 1
        return out_

    def _mode_cycle(self, out_, f_):
        if self.mode == 1:
            for i in range(len(self.extractors_down)):
                out_, f_, out_list, n_out_list, f_list = self._cycle_down(
                    self.extractors_down[i], i, self.layers, out_, f_
                )
                print(out_.size())
                if i < len(self.extractors_up) + 1:
                    out_ = self._cycle_up(
                        self.extractors_up[i],
                        i + 1,
                        self.layers,
                        out_,
                        out_list,
                        n_out_list,
                        f_list,
                    )
                print(out_.size())
                f_ = out_
        elif self.mode == 3:
            for i in range(len(self.extractors_down)):
                out_, f_, out_list, n_out_list, f_list = self._cycle_down(
                    self.extractors_down[i], i + 1, out_, f_
                )
                if i < len(self.extractors_up):
                    out_ = self._cycle_up(
                        self.extractors_up[i], i + 1, out_, out_list, n_out_list, f_list
                    )
        elif self.mode == 2:
            out_, f_, out_list, n_out_list, f_list = self._cycle_down(
                self.extractors_down[0], self.layers, out_, f_
            )
            for i in range(len(self.extractors_down) - 1):
                out_, f_, out_list, n_out_list, f_list = self._cycle_down(
                    self.extractors_down[i], i + 1, out_, f_
                )
                if i < len(self.extractors_up):
                    out_ = self._cycle_up(
                        self.extractors_up[i], i + 1, out_, out_list, n_out_list, f_list
                    )

        return out_, f_

    def _forward_impl(self, x):

        out = self.conv0(x)
        out = self.bn0(out)
        out = self.activation0(out)
        out = self.maxpool(out)
        f_ = out
        out_ = out * 0

        out_, f_ = self._mode_cycle(out_, f_)

        out = out_
        out = self.global_maxpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        out *= self.scale

        return out
