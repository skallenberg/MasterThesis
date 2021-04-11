from datetime import datetime

import torch

from ignite.contrib.handlers.tensorboard_logger import *
from ignite.engine import Events

from utils.config import Config
from utils import set_trainer

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

tb_metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-Score",
    "Top-3_Error",
    "Top-5_Error",
]


def train(net, dataset):

    config = Config().get_instance()

    if torch.cuda.device_count() > 1:
        writer_name = (
            net.module.name
            + "_"
            + dataset.name
            + "_"
            + datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        )
        net.module.writer = writer_name
    else:
        writer_name = (
            net.name + "_" + dataset.name + "_" + datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        )
        net.writer = writer_name
        
### examples for tb_logger taken from https://pytorch.org/ignite/v0.4.2/contrib/handlers.html#tensorboard-logger

    tb_logger = TensorboardLogger(log_dir="./data/models/logs/runs/" + writer_name)

    trainer, test_evaluator = set_trainer.get_trainer(
        net,
        dataset,
        early_stop=config["Trainer"]["EarlyStopping"],
        scheduler=config["Trainer"]["LRScheduler"],
        lrfinder=config["Trainer"]["LRFinder"],
    )

    tb_logger.attach_output_handler(
        trainer,
        event_name=Events.ITERATION_COMPLETED,
        tag="training",
        output_transform=lambda loss: {"Loss": loss},
    )
    
    tb_logger.attach(trainer,
                 log_handler=OutputHandler(tag="training", metric_names="all", global_step_transform=global_step_from_engine(trainer)),
                 event_name=Events.ITERATION_COMPLETED,
                 )

    tb_logger.attach_output_handler(
        test_evaluator,
        event_name=Events.EPOCH_COMPLETED,
        tag="validation",
        metric_names=tb_metrics,
        global_step_transform=global_step_from_engine(trainer),
    )

    with torch.autograd.set_detect_anomaly(True):
        trainer.run(dataset.trainloader, max_epochs=config["Setup"]["Epochs"])

    tb_logger.close()
