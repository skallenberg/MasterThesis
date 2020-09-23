import torch
import torch.nn as nn
import torch.nn.functional as F

from model.common import *

from .blocks import *
from .utils import *

from utils.config import Config

config = Config.get_instance()

data_name = config["Setup"]["Data"]
alpha = config["Misc"]["CELU_alpha"]