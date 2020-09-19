from datetime import datetime

import torch

from ignite.contrib.handlers.tensorboard_logger import *
from ignite.engine import Events


from utils import set_trainer
from utils.config import Config

config = Config.get_instance()

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

tb_metrics = [
    "Accuracy",
    "Running_Average_Accuracy",
    "Loss",
    "Running_Average_Loss",
    "Precision",
    "Recall",
    "F1-Score",
    "ROC_AUC",
]


def train(net, dataset):

    if torch.cuda.device_count() > 1:
        writer_name = (
            net.module.name
            + "_"
            + dataset.name
            + "_"
            + datetime.now().strftime("%m-%d-%Y-%H:%M:%S")
        )
        net.module.writer = writer_name
    else:
        writer_name = (
            net.name + "_" + dataset.name + "_" + datetime.now().strftime("%m-%d-%Y-%H:%M:%S")
        )
        net.writer = writer_name

    tb_logger = TensorboardLogger(log_dir="./data/models/logs/runs/" + writer_name)

    trainer, train_evaluator, test_evaluator = set_trainer.get_trainer(
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

    tb_logger.attach_output_handler(
        train_evaluator,
        event_name=Events.EPOCH_COMPLETED,
        tag="training_validation",
        metric_names=tb_metrics,
        global_step_transform=global_step_from_engine(trainer),
    )

    tb_logger.attach_output_handler(
        test_evaluator,
        event_name=Events.EPOCH_COMPLETED,
        tag="validation",
        metric_names=tb_metrics,
        global_step_transform=global_step_from_engine(trainer),
    )

    tb_logger.attach(
        trainer,
        event_name=Events.ITERATION_COMPLETED(every=500),
        log_handler=WeightsScalarHandler(net),
    )

    tb_logger.attach(
        trainer, event_name=Events.EPOCH_COMPLETED, log_handler=WeightsHistHandler(net),
    )

    tb_logger.attach(
        trainer,
        event_name=Events.ITERATION_COMPLETED(every=500),
        log_handler=GradsScalarHandler(net),
    )

    tb_logger.attach(trainer, event_name=Events.EPOCH_COMPLETED, log_handler=GradsHistHandler(net))

    with torch.autograd.set_detect_anomaly(True):
        trainer.run(dataset.trainloader, max_epochs=config["Setup"]["Epochs"])

    tb_logger.close()
