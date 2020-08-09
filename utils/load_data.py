import torch
import torchvision
import torchvision.transforms as transforms
from utils.config import Config

config = Config.get_instance()

data_name = config["Setup"]["Data"]


class dataset:
    def __init__(self, name, trainloader, testloader, classes):
        self.name = name
        self.trainloader = trainloader
        self.testloader = testloader
        self.classes = classes


def get_data(augment=False):
    if data_name == "cifar10":
        mean = [0.49139968, 0.48215827, 0.44653124]
        std = [0.24703233, 0.24348505, 0.26158768]
        if augment:
            tr_transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )
        else:
            tr_transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean, std),]
            )

        val_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std),]
        )
        trainset = torchvision.datasets.CIFAR10(
            root="./data/datasets", train=True, download=True, transform=tr_transform
        )

        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=64, shuffle=True, num_workers=2
        )

        testset = torchvision.datasets.CIFAR10(
            root="./data/datasets", train=False, download=True, transform=val_transform
        )

        testloader = torch.utils.data.DataLoader(
            testset, batch_size=64, shuffle=False, num_workers=2
        )

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
    elif data_name == "cifar100":
        mean = [0.50707516, 0.48654887, 0.44091784]
        std = [0.26733429, 0.25643846, 0.27615047]
        if augment:
            tr_transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )
        else:
            tr_transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean, std),]
            )

        val_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std),]
        )

        trainset = torchvision.datasets.CIFAR100(
            root="./data/datasets", train=True, download=True, transform=tr_transform
        )

        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=64, shuffle=True, num_workers=2
        )

        testset = torchvision.datasets.CIFAR100(
            root="./data/datasets", train=False, download=True, transform=val_transform
        )

        testloader = torch.utils.data.DataLoader(
            testset, batch_size=64, shuffle=False, num_workers=2
        )

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

    return dataset(data_name, trainloader, testloader, classes)
