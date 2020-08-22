import logging
import time
from datetime import datetime

import ignite.metrics as metrics
import torch
import torch.nn as nn
from ignite.contrib.handlers.param_scheduler import CosineAnnealingScheduler
from ignite.contrib.handlers.tensorboard_logger import *
from ignite.engine import Events
from ignite.engine import create_supervised_evaluator
from ignite.engine import create_supervised_trainer
from torch.utils.tensorboard import SummaryWriter

from utils import set_config
from utils import visualize
from utils.config import Config

config = Config.get_instance()

if torch.cuda.is_available():
    device = torch.device(config["Setup"]["Device"])
    if torch.cuda.device_count() > 1:
        config["Setup"]["Parallel"] = 1
    else:
        config["Setup"]["Parallel"] = 0
else:
    device = torch.device("cpu")
    config["Setup"]["Parallel"] = 0


def train(net, dataset):

    if config["Setup"]["Parallel"] == 1:
        writer_name = (
            net.module.name
            + "_"
            + dataset.name
            + "_"
            + datetime.now().strftime("%m-%d-%Y-%H:%M:%S")
        )
    else:
        writer_name = (
            net.name + "_" + dataset.name + "_" + datetime.now().strftime("%m-%d-%Y-%H:%M:%S")
        )

    tb_logger = TensorboardLogger(log_dir="./data/models/log/runs/" + writer_name)
    net.module.writer = writer_name

    optimizer = set_config.choose_optimizer(net)
    criterion = nn.CrossEntropyLoss()

    trainer = create_supervised_trainer(net, optimizer, criterion, device=device)

    val_metrics = {
        "Accuracy": metrics.Accuracy(),
        "Loss": metrics.Loss(criterion),
        "Precision": metrics.Precision(average=True),
        "Recall": metrics.Recall(average=True),
        "F1-Score": metrics.Fbeta(beta=1.0),
        "Confusion Matrix": metrics.ConfusionMatrix(num_classes=len(dataset.classes)),
    }

    train_evaluator = create_supervised_evaluator(net, metrics=val_metrics, device=device)
    test_evaluator = create_supervised_evaluator(net, metrics=val_metrics, device=device)

    scheduler = CosineAnnealingScheduler(optimizer, "lr", 1e-1, 1e-3, len(dataset.trainloader))
    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

    tb_logger.attach_output_handler(
        trainer,
        event_name=Events.ITERATION_COMPLETED,
        tag="training",
        output_transform=lambda loss: {"Loss": loss},
    )

    tb_logger.attach_output_handler(
        train_evaluator,
        event_name=Events.EPOCH_COMPLETED,
        tag="training",
        metric_names=["Accuracy", "Loss", "Precision", "Recall", "F1-Score"],
        global_step_transform=global_step_from_engine(trainer),
    )

    tb_logger.attach_output_handler(
        test_evaluator,
        event_name=Events.EPOCH_COMPLETED,
        tag="validation",
        metric_names=["Accuracy", "Loss", "Precision", "Recall", "F1-Score"],
        global_step_transform=global_step_from_engine(trainer),
    )

    tb_logger.attach(
        trainer, event_name=Events.ITERATION_COMPLETED, log_handler=WeightsScalarHandler(net),
    )

    tb_logger.attach(
        trainer, event_name=Events.EPOCH_COMPLETED, log_handler=WeightsHistHandler(net),
    )

    tb_logger.attach(
        trainer, event_name=Events.ITERATION_COMPLETED, log_handler=GradsScalarHandler(net),
    )

    tb_logger.attach(trainer, event_name=Events.EPOCH_COMPLETED, log_handler=GradsHistHandler(net))

    @trainer.on(Events.ITERATION_COMPLETED(every=config["Trainer"]["log_interval"]))
    def log_training_metrics(engine):
        logging.info("Epoch[{}] Loss: {:.2f}".format(engine.state.epoch, engine.state.output))

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        train_evaluator.run(dataset.trainloader)
        metrics = train_evaluator.state.metrics
        logging.info(
            "Training Results - Epoch:\t{}\nAverage Accuracy:\t{:.2f}\nAverage Loss:\t{:.2f}\nAverage Precision:\t{:.2f}\nAverage Recall:\t{:.2f}\nF1-Score:\t{:.2f}".format(
                engine.state.epoch,
                metrics["Accuracy"],
                metrics["Loss"],
                metrics["Precision"],
                metrics["Recall"],
                metrics["F1-Score"],
            )
        )

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        test_evaluator.run(dataset.testloader)
        metrics = test_evaluator.state.metrics
        logging.info(
            "Validation Results - Epoch:\t{}\nAccuracy:\t{:.2f}\nLoss:\t{:.2f}\nPrecision:\t{:.2f}\nRecall:\t{:.2f}\nF1-Score:\t{:.2f}".format(
                engine.state.epoch,
                metrics["Accuracy"],
                metrics["Loss"],
                metrics["Precision"],
                metrics["Recall"],
                metrics["F1-Score"],
            )
        )
        logging.info("Confusion Matrix:\n", metrics["Confusion Matrix"])

    trainer.run(dataset.trainloader, max_epochs=100)
