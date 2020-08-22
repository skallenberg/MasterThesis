import logging

import torch
import torch.nn as nn

import mode
from utils import load_data
from utils import set_config
from utils import visualize
from utils.config import Config

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

config = Config.get_instance()

logging.basicConfig(level=logging.INFO)

data = load_data.get_data(augment=config["DataLoader"]["Augment"])

logging.info("Loaded Dataset")

if torch.cuda.is_available():
    device = torch.device(config["Setup"]["Device"])
    logging.info("Running on %i GPUs" % (torch.cuda.device_count()))
else:
    device = torch.device("cpu")
    logging.info("Running on CPU")

net = set_config.choose_architecture()

if torch.cuda.device_count() > 1:
    net = nn.DistributedDataParallel(net)
    config["Setup"]["Parallel"] = 1
else:
    config["Setup"]["Parallel"] = 0
net = net.to(device)


logging.info("Initialized Network")

# visualize.make_graph(net)

net = mode.train(net, data)

logging.info("Finished Training")

# utils.visualize.show(data, nimages=4, net=net)

mode.test(net, data)

logging.info("Finished Testing")
logging.info("Run succesfull")
