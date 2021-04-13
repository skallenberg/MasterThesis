import logging

import torch
import torch.nn as nn

import ignite_training
from utils import load_data
from utils import set_config
from utils.config import Config

logging.basicConfig(level=logging.INFO)

if torch.cuda.is_available():
    device = torch.device("cuda")
    logging.info("Running on %i GPUs" % (torch.cuda.device_count()))
else:
    device = torch.device("cpu")
    logging.info("Running on CPU")


def train_model():
    config = Config().get_instance()
    torch.backends.cudnn.enabled = config["Misc"]["cudnnEnabled"]
    torch.backends.cudnn.benchmark = config["Misc"]["cudnnBenchmark"]

    data = load_data.get_data()

    logging.info("Loaded Dataset")

    net = set_config.choose_architecture()

    net = net.to(device)

    logging.info("Initialized Network")

    net = ignite_training.train(net, data)

    logging.info("Finished Run")

    logging.info("Run successfull")

if __name__ == "__main__":
    train_model()