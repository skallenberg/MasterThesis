import logging

import torch
import torch.nn as nn
import tomlkit

import ignite_training
from utils import load_data, set_config, visualize
from utils.config import Config

config = Config().get_instance()

torch.backends.cudnn.enabled = config["Misc"]["cudnnEnabled"]
torch.backends.cudnn.benchmark = config["Misc"]["cudnnBenchmark"]

logging.basicConfig(level=logging.INFO)

net = set_config.choose_architecture()
print(net.name)
config["Setup"]["Architecture"] = "ResBasenet"

# data = config.__str__()
# with open("config.toml", "w+") as f:
#    f.write(data)
# f.close()

net = set_config.choose_architecture()
print(net.name)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    logging.info("Running on %i GPUs" % (torch.cuda.device_count()))
else:
    device = torch.device("cpu")
    logging.info("Running on CPU")

data = load_data.get_data()

logging.info("Loaded Dataset")

net = set_config.choose_architecture()


if torch.cuda.device_count() > 1:
    net = nn.DistributedDataParallel(net)
net = net.to(device)


logging.info("Initialized Network")

# visualize.make_graph(net)

net = ignite_training.train(net, data)

logging.info("Finished Training")

# utils.visualize.show(data, nimages=4, net=net)

# mode.test(net, data)

logging.info("Finished Testing")
logging.info("Run succesfull")
