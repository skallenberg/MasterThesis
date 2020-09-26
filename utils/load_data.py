from functools import partial

import numpy as np
import torch

from utils.config import Config
from utils.data_utils import *


class dataset:
    def __init__(self, name, trainloader, testloader, classes, train_set=None, valid_set=None):
        self.name = name
        self.trainloader = trainloader
        self.testloader = testloader
        self.classes = classes
        self.train_set = train_set
        self.valid_set = valid_set


def get_data(return_sets=False):
    config = Config().get_instance()

    data_name = config["Setup"]["Data"]
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
                data["train"]["data"].numpy(), axis=(0, 1, 2)
            ),  # equals np.mean(cifar10()['train']['data'], axis=(0,1,2))
            np.std(
                data["train"]["data"].numpy(), axis=(0, 1, 2)
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

    if return_sets:
        dataset_final = dataset(
            data_name,
            train_batches,
            valid_batches,
            classes,
            train_set=train_set,
            valid_set=valid_set,
        )
    else:
        dataset_final = dataset(data_name, train_batches, valid_batches, classes)

    return dataset_final
