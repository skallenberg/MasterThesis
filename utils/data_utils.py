import copy
import os
from collections import defaultdict
from collections import namedtuple
from functools import lru_cache as cache
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torchvision

from utils.config import Config

gpu_path = "/data/skallenberg/datasets"
local_path = "./data/datasets"

if os.path.isdir(gpu_path):
    data_path = gpu_path
else:
    data_path = local_path

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

cpu = torch.device("cpu")


### Helper Functions


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def flip_lr(x):
    if isinstance(x, torch.Tensor):
        return torch.flip(x, [-1])
    return x[..., ::-1].copy()


def map_nested(func, nested_dict):
    return {
        k: map_nested(func, v) if isinstance(v, dict) else func(v) for k, v in nested_dict.items()
    }


def preprocess(dataset, transforms):
    dataset = copy.copy(dataset)
    for transform in reversed(transforms):
        dataset["data"] = transform(dataset["data"])
    return dataset


### Transformations

chunks = lambda data, splits: (data[start:end] for (start, end) in zip(splits, splits[1:]))


even_splits = lambda N, num_chunks: np.cumsum(
    [0]
    + [(N // num_chunks) + 1] * (N % num_chunks)
    + [N // num_chunks] * (num_chunks - (N % num_chunks))
)


def shuffled(xs, inplace=False):
    xs = xs if inplace else copy.copy(xs)
    np.random.shuffle(xs)
    return xs


def transformed(data, targets, transform, max_options=None, unshuffle=False):
    i = torch.randperm(len(data), device=device)
    data = data[i]
    options = shuffled(transform.options(data.shape), inplace=True)[:max_options]
    data = torch.cat(
        [
            transform.apply(x, **choice)
            for choice, x in zip(options, chunks(data, even_splits(len(data), len(options))))
        ]
    )
    return (data[torch.argsort(i)], targets) if unshuffle else (data, targets[i])


##### Classes


class Batches:
    def __init__(
        self,
        batch_size,
        transforms=(),
        dataset=None,
        shuffle=True,
        drop_last=False,
        max_options=None,
    ):
        self.dataset, self.transforms, self.shuffle, self.max_options = (
            dataset,
            transforms,
            shuffle,
            max_options,
        )
        N = len(dataset["data"])
        self.splits = list(range(0, N + 1, batch_size))
        if not drop_last and self.splits[-1] != N:
            self.splits.append(N)

    def __iter__(self):
        data, targets = self.dataset["data"], self.dataset["targets"]
        for transform in self.transforms:
            data, targets = transformed(
                data, targets, transform, max_options=self.max_options, unshuffle=not self.shuffle
            )
        if self.shuffle:
            i = torch.randperm(len(data), device=device)
            data, targets = data[i], targets[i]
        return (
            {"input": x.clone(), "target": y}
            for (x, y) in zip(chunks(data, self.splits), chunks(targets, self.splits))
        )

    def __len__(self):
        return len(self.splits) - 1


class Crop(namedtuple("Crop", ("h", "w"))):
    def apply(self, x, x0, y0):
        return x[..., y0 : y0 + self.h, x0 : x0 + self.w]

    def options(self, shape):
        *_, H, W = shape
        return [
            {"x0": x0, "y0": y0} for x0 in range(W + 1 - self.w) for y0 in range(H + 1 - self.h)
        ]


class FlipLR(namedtuple("FlipLR", ())):
    def apply(self, x, choice):
        return flip_lr(x) if choice else x

    def options(self, shape):
        return [{"choice": b} for b in [True, False]]


class Cutout(namedtuple("Cutout", ("h", "w"))):
    def apply(self, x, x0, y0):
        x[..., y0 : y0 + self.h, x0 : x0 + self.w] = 0.0
        return x

    def options(self, shape):
        *_, H, W = shape
        return [
            {"x0": x0, "y0": y0} for x0 in range(W + 1 - self.w) for y0 in range(H + 1 - self.h)
        ]


######## Dataset Download Functions


@cache(None)
def cifar10(root=data_path):
    download = lambda train: torchvision.datasets.CIFAR10(root=root, train=train, download=True)
    return {
        k: {"data": torch.tensor(v.data), "targets": torch.tensor(v.targets)}
        for k, v in [("train", download(True)), ("valid", download(False))]
    }


@cache(None)
def cifar100(root=data_path):
    download = lambda train: torchvision.datasets.CIFAR100(root=root, train=train, download=True)
    return {
        k: {"data": torch.tensor(v.data), "targets": torch.tensor(v.targets),}
        for k, v in [("train", download(True)), ("valid", download(False))]
    }


@cache(None)
def mnist(root=data_path):
    download = lambda train: torchvision.datasets.MNIST(root=root, train=train, download=True)
    return {
        k: {"data": torch.tensor(v.data), "targets": torch.tensor(v.targets)}
        for k, v in [("train", download(True)), ("valid", download(False))]
    }


@cache(None)
def svhn(root=data_path):
    download = lambda train: torchvision.datasets.SVHN(root=root, train=train, download=True)
    return {
        k: {"data": torch.tensor(v.data), "targets": torch.tensor(v.targets)}
        for k, v in [("train", download(True)), ("valid", download(False))]
    }
