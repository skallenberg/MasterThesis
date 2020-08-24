import torch
import torch.nn as nn
import copy
import os
from collections import namedtuple, defaultdict
from functools import partial
import numpy as np
import torchvision
from utils.config import Config
from functools import lru_cache as cache

config = Config.get_instance()

data_name = config["Setup"]["Data"]

worker_count = torch.cuda.device_count() * 4

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
        k: {"data": torch.tensor(v.data), "targets": torch.tensor(v.targets)}
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


class dataset:
    def __init__(self, name, trainloader, testloader, classes):
        self.name = name
        self.trainloader = trainloader
        self.testloader = testloader
        self.classes = classes


def get_data():
    if data_name == "cifar10":
        data = cifar10()
        cifar10_mean, cifar10_std = [
            (125.31, 122.95, 113.87),  # equals np.mean(cifar10()['train']['data'], axis=(0,1,2))
            (62.99, 62.09, 66.70),  # equals np.std(cifar10()['train']['data'], axis=(0,1,2))
        ]

        mean, std = [
            torch.tensor(x, device=device, dtype=torch.float32)
            for x in (cifar10_mean, cifar10_std)
        ]
        classes = (
            "plane",
            "car",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        )
    else:
        if data_name == "cifar100":
            data = cifar100()
            classes = (
                "beaver",
                "dolphin",
                "otter",
                "seal",
                "whale",
                "aquarium fish",
                "flatfish",
                "ray",
                "shark",
                "trout",
                "orchids",
                "poppies",
                "roses",
                "sunflowers",
                "tulips",
                "bottles",
                "bowls",
                "cans",
                "cups",
                "plates",
                "apples",
                "mushrooms",
                "oranges",
                "pears",
                "sweet peppers",
                "clock",
                "computer keyboard",
                "lamp",
                "telephone",
                "television",
                "bed",
                "chair",
                "couch",
                "table",
                "wardrobe",
                "bee",
                "beetle",
                "butterfly",
                "caterpillar",
                "cockroach",
                "bear",
                "leopard",
                "lion",
                "tiger",
                "wolf",
                "bridge",
                "castle",
                "house",
                "road",
                "skyscraper",
                "cloud",
                "forest",
                "mountain",
                "plain",
                "sea",
                "camel",
                "cattle",
                "chimpanzee",
                "elephant",
                "kangaroo",
                "fox",
                "porcupine",
                "possum",
                "raccoon",
                "skunk",
                "crab",
                "lobster",
                "snail",
                "spider",
                "worm",
                "baby",
                "boy",
                "girl",
                "man",
                "woman",
                "crocodile",
                "dinosaur",
                "lizard",
                "snake",
                "turtle",
                "hamster",
                "mouse",
                "rabbit",
                "shrew",
                "squirrel",
                "maple",
                "oak",
                "palm",
                "pine",
                "willow",
                "bicycle",
                "bus",
                "motorcycle",
                "pickup truck",
                "train",
                "lawn-mower",
                "rocket",
                "streetcar",
                "tank",
                "tractor",
            )
        elif data_name == "svhn":
            data = svhn()
            classes = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")
        elif data_name == "mnist":
            data = mnist()
            classes = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")
        data_mean, data_std = [
            np.mean(
                dataset["train"]["data"], axis=(0, 1, 2)
            ),  # equals np.mean(cifar10()['train']['data'], axis=(0,1,2))
            np.std(
                dataset["train"]["data"], axis=(0, 1, 2)
            ),  # equals np.std(cifar10()['train']['data'], axis=(0,1,2))
        ]

        mean, std = [
            torch.tensor(x, device=device, dtype=torch.float32) for x in (data_mean, data_std)
        ]

    normalise = lambda data, mean=mean, std=std: (data - mean) / std
    unnormalise = lambda data, mean=mean, std=std: data * std + mean
    pad = lambda data, border: nn.ReflectionPad2d(border)(data)
    transpose = lambda x, source="NHWC", target="NCHW": x.permute(
        [source.index(d) for d in target]
    )
    to = lambda *args, **kwargs: (lambda x: x.to(*args, **kwargs))

    data = map_nested(to(device), data)
    train_set = preprocess(
        data["train"], [partial(pad, border=4), transpose, normalise, to(torch.float32)]
    )
    valid_set = preprocess(data["valid"], [transpose, normalise, to(torch.float32)])
    train_batcher = partial(
        Batches, dataset=train_set, shuffle=True, drop_last=True, max_options=200
    )
    valid_batcher = partial(Batches, dataset=valid_set, shuffle=False, drop_last=False)
    train_batches = train_batcher(
        batch_size=config["DataLoader"]["BatchSize"],
        transforms=(Crop(32, 32), FlipLR(), Cutout(8, 8)),
    )
    valid_batches = valid_batcher(batch_size=config["DataLoader"]["BatchSize"])

    return_set = dataset(data_name, train_batches, valid_batches, classes)

    return return_set
