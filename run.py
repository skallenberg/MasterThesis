from utils import load_data, set_config, visualize
import mode
import torch
import logging

logging.basicConfig(level=logging.INFO)

data = load_data.get_data()

logging.info("Loaded Dataset")

if torch.cuda.is_available():
    device = torch.device("cuda:1")
    logging.info("Running on the GPU")
else:
    device = torch.device("cpu")
    logging.info("Running on the CPU")

net = set_config.choose_architecture()

net = net.to(device)

logging.info("Initialized Network")

# visualize.make_graph(net)

net = mode.train(net, data)

logging.info("Finished Training")

# utils.visualize.show(data, nimages=4, net=net)

mode.test(net, data)

logging.info("Finished Testing")
logging.info("Run succesfull")

