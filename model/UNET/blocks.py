import torch
import torch.nn as nn
import torch.nn.functional as F

from model.common import *
from utils.config import Config

from .utils import *

config = Config.get_instance()

alpha = config["Misc"]["CELU_alpha"]
