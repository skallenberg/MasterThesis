import torch
import torch.nn as nn
import torch.functional as F


class stem_v2(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv0 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=0)
        self.bn0 = nn.BatchNorm2d(32)
        self.activation0 = nn.ReLU(inplace=True)

        self.l0 = nn.Sequential(self.conv0, self.bn0, self.activation0)

        self.conv1 = nn.Conv2d(32, 32, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.activation1 = nn.ReLU(inplace=True)

        self.l1 = nn.Sequential(self.conv1, self.bn1, self.activation1)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.activation2 = nn.ReLU(inplace=True)

        self.l2 = nn.Sequential(self.conv2, self.bn2, self.activation2)

        self.maxpool_0_0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.conv_1_0 = nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=0)

        self.conv_0_1 = nn.Conv2d(160, 64, kernel_size=1)
        self.conv_0_2 = nn.Conv2d(64, 96, kernel_size=3, padding=0)

        self.l0_1 = nn.Sequential(self.conv_0_1, self.conv_0_2)

        self.conv_1_1 = nn.Conv2d(160, 64, kernel_size=1, padding=1)
        self.conv_1_2 = nn.Conv2d(64, 64, kernel_size=(7, 1), padding=1)
        self.conv_1_3 = nn.Conv2d(64, 64, kernel_size=(1, 7), padding=1)
        self.conv_1_4 = nn.Conv2d(64, 96, kernel_size=3, padding=0)

        self.l1_1 = nn.Sequential(
            self.conv_1_1, self.conv_1_2, self.conv_1_3, self.conv_1_4
        )

        self.conv_0_3 = nn.Conv2d(192, 192, kernel_size=3, stride=2, padding=0)
        self.maxpool_1_5 = nn.MaxPool2d(stride=2, kernel_size=3, padding=0)

    def _forward_impl(self, x):
        out = self.l0(x)

        out = self.l1(out)

        out = self.l2(out)

        split_out = [out, out]

        split_out[0] = self.maxpool_0_0(out)
        split_out[1] = self.conv_1_0(out)

        out = torch.cat(split_out, dim=1)

        split_out = [out, out]

        split_out[0] = self.l0_1(out)
        split_out[1] = self.l1_1(out)

        out = torch.cat(split_out, dim=1)

        split_out = [out, out]

        split_out[0] = self.conv_0_3(out)
        split_out[1] = self.maxpool_1_5(out)

        out = torch.cat(split_out, dim=1)

        return out

    def forward(self, x):
        out = self._forward_impl(x)
        return out


class InceptionA(nn.Module):
    def __init__(self, channels_in):
        super().__init__()

        self.avg_0_0 = nn.AdaptiveAvgPool2d((35, 35))
        self.conv_0_1 = nn.Conv2d(channels_in, 96, kernel_size=1)

        self.conv_1_0 = nn.Conv2d(channels_in, 96, kernel_size=1)

        self.conv_2_0 = nn.Conv2d(channels_in, 64, kernel_size=1)
        self.conv_2_1 = nn.Conv2d(64, 96, kernel_size=3, padding=1)

        self.conv_3_0 = nn.Conv2d(channels_in, 64, kernel_size=1)
        self.conv_3_1 = nn.Conv2d(64, 96, kernel_size=3, padding=1)
        self.conv_3_2 = nn.Conv2d(96, 96, kernel_size=3, padding=1)

    def _forward_impl(self, x):
        split_out = [x, x, x, x]

        split_out[0] = self.conv_0_1(self.avg_0_0(x))
        split_out[1] = self.conv_1_0(x)
        split_out[2] = self.conv_2_1(self.conv_2_0(x))
        split_out[3] = self.conv_3_2(self.conv_3_1(self.conv_3_0(x)))

        out = torch.cat(split_out, 1)

        return out

    def forward(self, x):
        out = self._forward_impl(x)
        return out


class ReductionA(nn.Module):
    def __init__(self, channels_in, k, l, m, n):
        super().__init__()
        self.maxpool_0_0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        self.conv_1_0 = nn.Conv2d(channels_in, n, kernel_size=3, stride=2, padding=0)
        self.conv_2_0 = nn.Conv2d(channels_in, k, kernel_size=1)
        self.conv_2_1 = nn.Conv2d(k, l, kernel_size=3, padding=1)
        self.conv_2_2 = nn.Conv2d(l, m, kernel_size=3, stride=2, padding=0)

    def _forward_impl(self, x):
        split_out = [x, x, x]

        split_out[0] = self.maxpool_0_0(x)
        split_out[1] = self.conv_1_0(x)
        split_out[2] = self.conv_2_2(self.conv_2_1(self.conv_2_0(x)))

        out = torch.cat(split_out, 1)

        return out

    def forward(self, x):
        out = self._forward_impl(x)
        return out


class InceptionB(nn.Module):
    def __init__(self, channels_in):
        super().__init__()

        self.avg_0_0 = nn.AdaptiveAvgPool2d((17, 17))
        self.conv_0_1 = nn.Conv2d(channels_in, 128, kernel_size=1)

        self.conv_1_0 = nn.Conv2d(channels_in, 384, kernel_size=1)

        self.conv_2_0 = nn.Conv2d(channels_in, 192, kernel_size=1)
        self.conv_2_1 = nn.Conv2d(192, 224, kernel_size=(1, 7), padding=1)
        self.conv_2_2 = nn.Conv2d(224, 256, kernel_size=(7, 1), padding=2)

        self.conv_3_0 = nn.Conv2d(channels_in, 192, kernel_size=1)
        self.conv_3_1 = nn.Conv2d(192, 192, kernel_size=(1, 7), padding=1)
        self.conv_3_2 = nn.Conv2d(192, 224, kernel_size=(7, 1), padding=1)
        self.conv_3_3 = nn.Conv2d(224, 224, kernel_size=(1, 7), padding=2)
        self.conv_3_4 = nn.Conv2d(224, 256, kernel_size=(7, 1), padding=2)

    def _forward_impl(self, x):
        split_out = [x, x, x, x]

        split_out[0] = self.conv_0_1(self.avg_0_0(x))
        split_out[1] = self.conv_1_0(x)
        split_out[2] = self.conv_2_2(self.conv_2_1(self.conv_2_0(x)))
        split_out[3] = self.conv_3_4(
            self.conv_3_3(self.conv_3_2(self.conv_3_1(self.conv_3_0(x))))
        )

        out = torch.cat(split_out, 1)

        return out

    def forward(self, x):
        out = self._forward_impl(x)
        return out


class ReductionB(nn.Module):
    def __init__(self, channels_in):
        super().__init__()
        self.maxpool_0_0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        self.conv_1_0 = nn.Conv2d(channels_in, 192, kernel_size=1)
        self.conv_1_1 = nn.Conv2d(192, 192, kernel_size=3, stride=2, padding=0)

        self.conv_2_0 = nn.Conv2d(channels_in, 256, kernel_size=1)
        self.conv_2_1 = nn.Conv2d(256, 256, kernel_size=(1, 7), padding=1)
        self.conv_2_2 = nn.Conv2d(256, 320, kernel_size=(7, 1), padding=2)
        self.conv_2_3 = nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=0)

    def _forward_impl(self, x):
        split_out = [x, x, x]

        split_out[0] = self.maxpool_0_0(x)
        split_out[1] = self.conv_1_1(self.conv_1_0(x))
        split_out[2] = self.conv_2_3(self.conv_2_2(self.conv_2_1(self.conv_2_0(x))))

        out = torch.cat(split_out, 1)

        return out

    def forward(self, x):
        out = self._forward_impl(x)
        return out


class InceptionC(nn.Module):
    def __init__(self, channels_in):
        super().__init__()

        self.avg_0_0 = nn.AdaptiveAvgPool2d((8, 8))
        self.conv_0_1 = nn.Conv2d(channels_in, 256, kernel_size=1)

        self.conv_1_0 = nn.Conv2d(channels_in, 256, kernel_size=1)

        self.conv_2_0 = nn.Conv2d(channels_in, 384, kernel_size=1)
        self.conv_2_1_0 = nn.Conv2d(384, 256, kernel_size=(1, 3), padding=(0, 1))
        self.conv_2_1_1 = nn.Conv2d(384, 256, kernel_size=(3, 1), padding=(1, 0))

        self.conv_3_0 = nn.Conv2d(channels_in, 384, kernel_size=1)
        self.conv_3_1 = nn.Conv2d(384, 448, kernel_size=(1, 3), padding=(0, 1))
        self.conv_3_2 = nn.Conv2d(448, 512, kernel_size=(3, 1), padding=(1, 0))
        self.conv_3_3_0 = nn.Conv2d(512, 256, kernel_size=(3, 1), padding=(1, 0))
        self.conv_3_3_1 = nn.Conv2d(512, 256, kernel_size=(1, 3), padding=(0, 1))

    def _forward_impl(self, x):
        split_out = [x, x, x, x, x, x]

        split_out[0] = self.conv_0_1(self.avg_0_0(x))
        split_out[1] = self.conv_1_0(x)
        split_out[2] = self.conv_2_1_0(self.conv_2_0(x))
        split_out[3] = self.conv_2_1_1(self.conv_2_0(x))
        split_out[4] = self.conv_3_3_0(self.conv_3_2(self.conv_3_1(self.conv_3_0(x))))
        split_out[5] = self.conv_3_3_1(self.conv_3_2(self.conv_3_1(self.conv_3_0(x))))

        out = torch.cat(split_out, 1)

        return out

    def forward(self, x):
        out = self._forward_impl(x)
        return out

