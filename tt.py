import torch.nn as nn


class TestNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(*[nn.ReLU(), nn.BatchNorm2d(64)])
        # print(list(self.modules()))
        self._init_module()
        print(list(self.modules()))

    def _init_module(self):
        for m in self.modules():
            if isinstance(m, nn.ReLU):
                self.m = nn.CELU(0.3)
            if isinstance(m, nn.BatchNorm2d):
                self.m = nn.Conv2d(m.num_features, 128, 3)

    def forward(self, x):
        return self.layer(x)


a = TestNet()

