import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from model.common import *

from .blocks import *
from .utils import interpolate

from utils.config import Config


class MGNet(nn.Module):
    def __init__(self, name, layers, num_classes, smoothing_steps=1, batch_norm=False, channel_growth="constant_equal", constant_extractors=False):
        super().__init__()

        self.config = Config().get_instance()

        self.name = name
        self.writer = ""
        self.layers = layers
        self.channels_in = 64
        self.smoothing_steps = smoothing_steps
        self.batch_norm = batch_norm
        self.channel_growth = channel_growth
        self.constant_extractors = constant_extractors
        
        if channel_growth=="constant_equal":
            self.data_size = 512
            self.feature_size = 256
            self.growth_rate = 1
        elif channel_growth=="constant_pair":
            self.data_size = 512
            self.feature_size = 256
            self.growth_rate = 1
        elif channel_growth=="progressive":
            self.data_size = None
            self.feature_size = None
            self.growth_rate = 2

        self.init_layer = init_block(self.channels_in)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        if channel_growth != "progressive":
            self.init_conv = conv_3x3(64, self.data_size)
            self.init_conv_2 = conv_3x3(64, self.feature_size)
            self.fc = nn.Linear(self.feature_size, num_classes)
        else:
            self.init_conv = nn.Identity()
            self.init_conv_2 = nn.Identity()
            self.fc = nn.Linear(64 * (2 ** layers), num_classes)

        self.global_maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.scale = self.config["Misc"]["FC_Scale"]

        self.mappings = self._set_data_feature_mapper(ch_start=self.feature_size,ch_out=self.data_size,growth_rate=self.growth_rate)
        self.extractors = self._set_feature_extractors(ch_start=self.data_size,ch_out=self.feature_size,growth_rate=self.growth_rate, constant=self.constant_extractors)
        self.feature_interpolations = self._set_interpolations(ch_start=self.feature_size,growth_rate=self.growth_rate)
        self.data_interpolations = self._set_interpolations(ch_start=self.data_size,growth_rate=self.growth_rate)

        #self._init_modules()

    def _init_modules(self):
        if self.config["Misc"]["UseCELU"]:
            nonlinearity = "leaky_relu"
        else:
            nonlinearity = "relu"
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity=nonlinearity)
            #elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            #    if self.config["Misc"]["GhostBatchNorm"]:
            #        pass
            #    else:
            #        nn.init.constant_(m.weight, 1)
            #        nn.init.constant_(m.bias, 0)

    def _set_data_feature_mapper(self, ch_start, ch_out, growth_rate=2):
        mappings = nn.ModuleList()
        if ch_start:
            ch_in = ch_start
        else:
            ch_in = self.channels_in
        if not ch_out:
            prog=True
        else:
            prog=False
        for i in range(self.layers + 1):
            if prog:
                ch_out = ch_in
            mappings.append(mapping_block(channels_in=ch_in,channels_out=ch_out,  batch_norm=self.batch_norm))
            ch_in = ch_in * growth_rate
        return mappings

    def _set_interpolations(self, ch_start, growth_rate=2):
        interpolations = nn.ModuleList()
        if ch_start:
            ch_in = ch_start
        else:
            ch_in = self.channels_in
        for i in range(self.layers):
            interpolations.append(conv_1x1(ch_in, ch_in * growth_rate, stride=2, groups=ch_in))
            ch_in *= growth_rate
        return interpolations

    def _set_feature_extractors(self, ch_start, ch_out, growth_rate=2, constant=False):
        extractors = nn.ModuleList()
        if ch_start:
            ch_in = ch_start
        else:
            ch_in = self.channels_in
        if not ch_out:
            prog=True
        else:
            prog=False
        for i in range(self.layers):
            grid_extractors = nn.ModuleList()
            if prog:
                ch_out = ch_in
            if constant:
                extractor = extractor_block(channels_in=ch_in,channels_out=ch_out, batch_norm=self.batch_norm)
                for j in range(self.smoothing_steps):
                    grid_extractors.append(
                        extractor
                    )
            else: 
                for j in range(self.smoothing_steps):
                    grid_extractors.append(
                        extractor_block(channels_in=ch_in,channels_out=ch_out, batch_norm=self.batch_norm)
                    )
            extractors.append(grid_extractors)
            ch_in *= growth_rate
        return extractors

    def _forward_impl(self, x):
        out = self.init_layer(x)
        data_out = self.init_conv_2(out)
        out = self.init_conv(out)
        #out = self.maxpool(out)
        #print(out.shape)
        f_ = out
        out_ = torch.zeros_like(data_out)
        for i in range(self.layers):
            for j in range(self.smoothing_steps):
                out_ = out_ + self.extractors[i][j](f_ - self.mappings[i](out_))
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


class VMGNet(MGNet):
    def __init__(self, name, layers, num_classes, smoothing_steps=3, batch_norm=False):
        super().__init__(name, layers, num_classes, smoothing_steps, batch_norm)

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
        out = self.init_layer(x)
        out = self.maxpool(out)
        f_ = out
        out_ = torch.zeros_like(out)
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
    def __init__(self, name, layers, num_classes, smoothing_steps=3, mode=1, batch_norm=False,channel_growth="constant_equal", interpol_type="interpol", constant_extractors=False):
        super().__init__(name, layers, num_classes, smoothing_steps=smoothing_steps, batch_norm=batch_norm, channel_growth=channel_growth, constant_extractors=constant_extractors)
        self.mode = mode
        if mode == 1 or mode == 2:
            self.n_extractor_steps = 2 * layers - 1
        else:
            self.n_extractor_steps = 2 * layers + 1

        self.prolongations = self._set_prolongation(interpol_type)
        self.extractors_down = nn.ModuleList([self.extractors])
        self.extractors_up = nn.ModuleList()
        for i in range(self.n_extractor_steps // 2):
            self.extractors_down.append(self._set_feature_extractors(ch_start=self.data_size,ch_out=self.feature_size,growth_rate=self.growth_rate))
            self.extractors_up.append(self._set_feature_extractors(ch_start=self.data_size,ch_out=self.feature_size,growth_rate=self.growth_rate))

    def _set_prolongation(self,interpol_type):
        prolongations = nn.ModuleList()
        ch_in = self.channels_in
        for i in range(1, self.layers + 1):
            if self.channel_growth != "progressive":
                prolongations.append(
                    interpolate(type=interpol_type,in_size=int(32*2**(-i)), scale=2)
                )
            else:
                prolongations.append(
                    interpolate(type=interpol_type,in_size=int(32*2**(-i)),channels_in=self.channels_in * 2 ** i,channel_scale=0.5, scale=2)
                )
        return prolongations

    def _mode_1_cycle(self, data_out, out):
        f_ = out
        out_ = torch.zeros_like(data_out)
        out_list = []
        n_out_list = []
        f_list = [f_]

        for k in range(self.layers):
            for i in range(k, self.layers):
                for j in range(self.smoothing_steps):
                    b = f_ - self.mappings[i](out_)
                    a = self.extractors_down[k][i][j](b)
                    out_ = out_ + a
                out_list.append(out_)
                n_out_ = self.feature_interpolations[i](out_)
                n_out_list.append(n_out_)
                a = self.mappings[i + 1](n_out_)
                b = self.data_interpolations[i](f_ - self.mappings[i](out_))
                f_ = a + b
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

    def _mode_2_cycle(self, data_out,out):
        f_ = out
        out_ = torch.zeros_like(data_out)
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

            if k > 0:
                for i in reversed(range(0, self.layers - k)):
                    a = out_ - n_out_list[i]
                    out_ = out_list[i] + self.prolongations[i](a)
                    for j in range(self.smoothing_steps):
                        out_ = out_ + self.extractors_up[len(self.extractors_up) - k][i][j](
                            f_list[i] - self.mappings[i](out_)
                        )

                f_ = f_list[0]
                out_list = []
                n_out_list = []
                f_list = [f_]
        return out_, f_

    def _mode_3_cycle(self, data_out,out):
        f_ = out
        out_ = torch.zeros_like(data_out)
        out_list = []
        n_out_list = []
        f_list = [f_]

        for i in range(self.layers):
            for j in range(self.smoothing_steps):
                out_ = out_ + self.extractors_down[0][i][j](f_ - self.mappings[i](out_))
            out_list.append(out_)
            n_out_ = self.feature_interpolations[i](out_)
            n_out_list.append(n_out_)
            f_ = self.data_interpolations[i](f_ - self.mappings[i](out_)) + self.mappings[i + 1](
                n_out_
            )
            f_list.append(f_)
            out_ = n_out_

        for k in reversed(range(self.layers)):
            for i in reversed(range(0, self.layers - k)):
                a = self.prolongations[i](out_ - n_out_list[i])
                out_ = out_list[i] + a
                for j in range(self.smoothing_steps):
                    out_ = out_ + self.extractors_up[len(self.extractors_up) - k][i][j](
                        f_list[i] - self.mappings[i](out_)
                    )

            f_ = f_list[0]
            out_list = []
            n_out_list = []
            f_list = [f_]

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

        return out_, f_

    def _cycle(self, data_out,out):
        if self.mode == 1:
            return self._mode_1_cycle(data_out,out)
        elif self.mode == 2:
            return self._mode_2_cycle(data_out,out)
        elif self.mode == 3:
            return self._mode_3_cycle(data_out,out)

    def _forward_impl(self, x):
        out = self.init_layer(x)
        #out = self.maxpool(out)
        out = self.init_layer(x)
        data_out = self.init_conv_2(out)
        out = self.init_conv(out)

        out, f_ = self._cycle(data_out,out)

        out = self.global_maxpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        out *= self.scale

        return out

    def forward(self, x):
        out = self._forward_impl(x)
        return out

